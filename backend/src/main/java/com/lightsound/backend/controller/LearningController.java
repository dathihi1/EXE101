package com.lightsound.backend.controller;

import com.lightsound.backend.model.*;
import com.lightsound.backend.repository.*;
import com.lightsound.backend.service.ApiTokenService;
import com.lightsound.backend.service.RecognitionGatewayService;
import com.lightsound.backend.service.VslCatalogService;
import com.lightsound.backend.service.VslCatalogService.Course;
import com.lightsound.backend.service.VslCatalogService.Sign;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.*;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/v1")
@CrossOrigin(origins = "*")
public class LearningController {
    private final UserRepository users;
    private final LearningProgressRepository progressRepository;
    private final PracticeAttemptRepository attemptRepository;
    private final DailyChallengeCompletionRepository challengeRepository;
    private final VslCatalogService catalog;
    private final ApiTokenService tokens;
    private final RecognitionGatewayService recognition;
    private final int learnedXp;
    private final int verifiedXp;
    private final int dailyXp;
    private final String recognitionPublicUrl;

    public LearningController(UserRepository users,
                              LearningProgressRepository progressRepository,
                              PracticeAttemptRepository attemptRepository,
                              DailyChallengeCompletionRepository challengeRepository,
                              VslCatalogService catalog,
                              ApiTokenService tokens,
                              RecognitionGatewayService recognition,
                              @Value("${vsl.rewards.learned-xp:20}") int learnedXp,
                              @Value("${vsl.rewards.verified-xp:30}") int verifiedXp,
                              @Value("${vsl.rewards.daily-xp:100}") int dailyXp,
                              @Value("${vsl.recognition.public-url:http://localhost:7860}") String recognitionPublicUrl) {
        this.users = users;
        this.progressRepository = progressRepository;
        this.attemptRepository = attemptRepository;
        this.challengeRepository = challengeRepository;
        this.catalog = catalog;
        this.tokens = tokens;
        this.recognition = recognition;
        this.learnedXp = learnedXp;
        this.verifiedXp = verifiedXp;
        this.dailyXp = dailyXp;
        this.recognitionPublicUrl = recognitionPublicUrl.replaceAll("/+$", "");
    }

    @PostMapping("/auth/guest")
    public GuestSession guest() {
        return new GuestSession(tokens.issueGuestToken(), true, LocalDateTime.now().plusDays(1).toEpochSecond(ZoneOffset.UTC));
    }

    @GetMapping("/learning/catalog")
    public CatalogView learningCatalog(@RequestHeader(value = "Authorization", required = false) String authorization) {
        Optional<User> user = optionalUser(authorization);
        Map<String, LearningProgress> progress = progressMap(user.orElse(null));
        List<CourseView> courses = new ArrayList<>();
        List<Course> source = catalog.courses();
        boolean previousCourseComplete = true;
        for (int courseIndex = 0; courseIndex < source.size(); courseIndex++) {
            Course course = source.get(courseIndex);
            boolean unlocked = courseIndex == 0 || previousCourseComplete;
            List<SignView> signs = buildSignViews(course, progress, unlocked);
            int learned = (int) signs.stream().filter(s -> s.learned || s.mastered).count();
            int mastered = (int) signs.stream().filter(s -> s.mastered).count();
            double completion = signs.isEmpty() ? 0.0 : (double) learned / signs.size();
            courses.add(new CourseView(course.getId(), course.getTitle(), course.getDescription(), course.getOrder(),
                    unlocked, course.getUnlockRatio(), completion, learned, mastered, signs));
            double required = courseIndex + 1 < source.size() ? source.get(courseIndex + 1).getUnlockRatio() : 1.0;
            previousCourseComplete = completion >= required;
        }
        return new CatalogView(1, courses);
    }

    @GetMapping("/signs")
    public SignListView signs(@RequestHeader(value = "Authorization", required = false) String authorization) {
        Map<String, LearningProgress> progress = progressMap(optionalUser(authorization).orElse(null));
        List<DictionarySignView> items = catalog.signs().stream().map(sign -> {
            LearningProgress item = progress.get(sign.getSignId());
            return new DictionarySignView(sign.getSignId(), sign.getDisplayName(), sign.getCourseId(), sign.getTip(),
                    recognitionPublicUrl + "/api/sample/" + sign.getSampleIndex(),
                    item != null && item.isLearned(), item != null && item.isMastered(),
                    item != null && item.isCameraVerified());
        }).toList();
        return new SignListView(items.size(), items);
    }

    @GetMapping("/progress/me")
    public ProgressSummary progress(@RequestHeader("Authorization") String authorization) {
        return buildSummary(requireUser(authorization));
    }

    @PostMapping("/progress/guest-import")
    public ProgressSummary importGuest(@RequestHeader("Authorization") String authorization,
                                       @RequestBody GuestImportRequest request) {
        User user = requireUser(authorization);
        if (request.items != null) {
            for (GuestProgressItem incoming : request.items) {
                Sign sign = catalog.findSign(incoming.signId)
                        .orElseThrow(() -> new ResponseStatusException(HttpStatus.BAD_REQUEST, "Từ không hợp lệ: " + incoming.signId));
                LearningProgress target = getOrCreate(user, sign);
                target.setQuizBestScore(Math.max(target.getQuizBestScore(), Math.max(0, incoming.quizBestScore)));
                target.setLearned(target.isLearned() || incoming.learned || target.getQuizBestScore() >= 2);
                target.setCameraVerified(target.isCameraVerified() || incoming.cameraVerified);
                target.setMastered(target.isLearned() && target.isCameraVerified());
                target.setXp(Math.max(target.getXp(), Math.max(0, incoming.xp)));
                target.setUpdatedAt(LocalDateTime.now());
                progressRepository.save(target);
            }
        }
        return buildSummary(user);
    }

    @PostMapping("/learning/progress/quiz")
    public ProgressSummary submitQuiz(@RequestHeader("Authorization") String authorization,
                                      @RequestBody QuizProgressRequest request) {
        User user = requireUser(authorization);
        Sign sign = catalog.findSign(request.signId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Không tìm thấy từ."));
        LearningProgress progress = getOrCreate(user, sign);
        int score = request.total <= 0 ? 0 : Math.min(3, Math.round((request.correct * 3f) / request.total));
        boolean wasLearned = progress.isLearned();
        progress.setQuizBestScore(Math.max(progress.getQuizBestScore(), score));
        if (progress.getQuizBestScore() >= 2) progress.setLearned(true);
        if (!wasLearned && progress.isLearned()) progress.setXp(progress.getXp() + learnedXp);
        progress.setMastered(progress.isLearned() && progress.isCameraVerified());
        progress.setUpdatedAt(LocalDateTime.now());
        progressRepository.save(progress);
        return buildSummary(user);
    }

    @PostMapping(value = "/practice/recognize", consumes = "multipart/form-data")
    public ResponseEntity<?> recognize(@RequestHeader("Authorization") String authorization,
                                       @RequestParam String expectedSignId,
                                       @RequestParam(required = false, defaultValue = "") String lessonId,
                                       @RequestPart("frames") List<MultipartFile> frames) {
        ApiTokenService.Session session = tokens.parseBearer(authorization)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Phiên đã hết hạn."));
        Sign sign = catalog.findSign(expectedSignId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Không tìm thấy từ."));
        Map<String, Object> result;
        try {
            result = new LinkedHashMap<>(recognition.recognize(expectedSignId, frames));
        } catch (IllegalArgumentException exception) {
            return ResponseEntity.badRequest().body(Map.of("message", exception.getMessage()));
        }

        int xpAwarded = 0;
        if (session.isUser()) {
            User user = users.findById(session.subject())
                    .orElseThrow(() -> new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Tài khoản không tồn tại."));
            boolean verified = Boolean.TRUE.equals(result.get("verified"));
            LearningProgress progress = getOrCreate(user, sign);
            boolean firstVerification = verified && !progress.isCameraVerified();
            if (verified) {
                progress.setCameraVerified(true);
                progress.setMastered(progress.isLearned() && progress.getQuizBestScore() >= 2);
                if (firstVerification) {
                    progress.setXp(progress.getXp() + verifiedXp);
                    xpAwarded = verifiedXp;
                }
                progress.setUpdatedAt(LocalDateTime.now());
                progressRepository.save(progress);
            }
            attemptRepository.save(PracticeAttempt.builder()
                    .user(user).courseId(sign.getCourseId()).signId(sign.getSignId())
                    .predictedSignId(String.valueOf(result.getOrDefault("predictedSignId", "")))
                    .status(String.valueOf(result.getOrDefault("status", "unknown")))
                    .confidence(asDouble(result.get("confidence"))).verified(verified).build());
            result.put("progress", buildSummary(user));
        }
        result.put("xpAwarded", xpAwarded);
        result.put("lessonId", lessonId);
        return ResponseEntity.ok(result);
    }

    @GetMapping("/challenges/today")
    public DailyChallengeView today(@RequestHeader(value = "Authorization", required = false) String authorization) {
        LocalDate today = LocalDate.now();
        List<Sign> all = catalog.signs();
        int offset = Math.floorMod(today.getDayOfYear() * 7, all.size());
        List<DictionarySignView> selected = new ArrayList<>();
        for (int i = 0; i < 3; i++) {
            Sign sign = all.get((offset + i * 9) % all.size());
            selected.add(new DictionarySignView(sign.getSignId(), sign.getDisplayName(), sign.getCourseId(), sign.getTip(),
                    recognitionPublicUrl + "/api/sample/" + sign.getSampleIndex(), false, false, false));
        }
        boolean completed = optionalUser(authorization)
                .flatMap(user -> challengeRepository.findByUserIdAndChallengeDate(user.getId(), today)).isPresent();
        return new DailyChallengeView(today.toString(), dailyXp, completed, selected);
    }

    @PostMapping("/challenges/today/complete")
    public ProgressSummary completeToday(@RequestHeader("Authorization") String authorization) {
        User user = requireUser(authorization);
        LocalDate today = LocalDate.now();
        challengeRepository.findByUserIdAndChallengeDate(user.getId(), today).orElseGet(() ->
                challengeRepository.save(DailyChallengeCompletion.builder()
                        .user(user).challengeDate(today).xpAwarded(dailyXp).build()));
        return buildSummary(user);
    }

    @GetMapping("/learning/leaderboard")
    public LeaderboardView leaderboard(@RequestParam(defaultValue = "weekly") String scope,
                                       @RequestParam(required = false) String courseId,
                                       @RequestHeader(value = "Authorization", required = false) String authorization) {
        LocalDateTime cutoff = "weekly".equals(scope) ? LocalDateTime.now().minusDays(7) : LocalDateTime.of(2000, 1, 1, 0, 0);
        Map<User, Integer> scores = new HashMap<>();
        for (LearningProgress progress : progressRepository.findAll()) {
            if (progress.getUpdatedAt().isBefore(cutoff)) continue;
            if (courseId != null && !courseId.isBlank() && !courseId.equals(progress.getCourseId())) continue;
            scores.merge(progress.getUser(), progress.getXp(), Integer::sum);
        }
        List<Map.Entry<User, Integer>> sorted = scores.entrySet().stream()
                .sorted(Map.Entry.<User, Integer>comparingByValue().reversed()
                        .thenComparing(entry -> entry.getKey().getName())).limit(100).toList();
        String currentUserId = optionalUser(authorization).map(User::getId).orElse("");
        List<LeaderboardEntry> entries = new ArrayList<>();
        for (int i = 0; i < sorted.size(); i++) {
            User user = sorted.get(i).getKey();
            entries.add(new LeaderboardEntry(i + 1, user.getName(), user.getOrgName(), sorted.get(i).getValue(),
                    user.getId().equals(currentUserId)));
        }
        return new LeaderboardView(scope, courseId == null ? "" : courseId, entries.size(), entries);
    }

    private List<SignView> buildSignViews(Course course, Map<String, LearningProgress> progress, boolean courseUnlocked) {
        List<SignView> result = new ArrayList<>();
        boolean previousLearned = true;
        for (Sign sign : course.getSigns()) {
            LearningProgress saved = progress.get(sign.getSignId());
            boolean learned = saved != null && saved.isLearned();
            boolean mastered = saved != null && saved.isMastered();
            boolean unlocked = courseUnlocked && previousLearned;
            String state = mastered ? "mastered" : learned ? "learned" : unlocked ? "available" : "locked";
            result.add(new SignView(sign.getSignId(), sign.getDisplayName(), sign.getTip(), sign.getOrder(), state,
                    unlocked, learned, mastered, saved != null && saved.isCameraVerified(),
                    saved == null ? 0 : saved.getQuizBestScore(),
                    recognitionPublicUrl + "/api/sample/" + sign.getSampleIndex()));
            previousLearned = learned;
        }
        return result;
    }

    private LearningProgress getOrCreate(User user, Sign sign) {
        return progressRepository.findByUserIdAndSignId(user.getId(), sign.getSignId()).orElseGet(() ->
                LearningProgress.builder().user(user).courseId(sign.getCourseId()).signId(sign.getSignId()).build());
    }

    private Map<String, LearningProgress> progressMap(User user) {
        if (user == null) return Map.of();
        return progressRepository.findByUserId(user.getId()).stream()
                .collect(Collectors.toMap(LearningProgress::getSignId, item -> item));
    }

    private Optional<User> optionalUser(String authorization) {
        return tokens.parseBearer(authorization).filter(ApiTokenService.Session::isUser)
                .flatMap(session -> users.findById(session.subject()));
    }

    private User requireUser(String authorization) {
        return optionalUser(authorization)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Cần đăng nhập để đồng bộ tiến độ."));
    }

    private ProgressSummary buildSummary(User user) {
        List<LearningProgress> items = progressRepository.findByUserId(user.getId());
        int learned = (int) items.stream().filter(LearningProgress::isLearned).count();
        int mastered = (int) items.stream().filter(LearningProgress::isMastered).count();
        int totalXp = items.stream().mapToInt(LearningProgress::getXp).sum()
                + challengeRepository.findByUserIdOrderByChallengeDateDesc(user.getId()).stream()
                .mapToInt(DailyChallengeCompletion::getXpAwarded).sum();
        Set<LocalDate> activityDays = items.stream().map(item -> item.getUpdatedAt().toLocalDate()).collect(Collectors.toSet());
        challengeRepository.findByUserIdOrderByChallengeDateDesc(user.getId()).forEach(item -> activityDays.add(item.getChallengeDate()));
        int streak = 0;
        LocalDate cursor = LocalDate.now();
        if (!activityDays.contains(cursor)) cursor = cursor.minusDays(1);
        while (activityDays.contains(cursor)) { streak++; cursor = cursor.minusDays(1); }
        List<RecentActivity> recent = attemptRepository.findTop20ByUserIdOrderByCreatedAtDesc(user.getId()).stream().limit(10)
                .map(attempt -> new RecentActivity(attempt.getSignId(), attempt.getStatus(), attempt.isVerified(), attempt.getCreatedAt().toString()))
                .toList();
        return new ProgressSummary(learned, mastered, totalXp, streak, items.size(), recent);
    }

    private double asDouble(Object value) {
        return value instanceof Number number ? number.doubleValue() : 0.0;
    }

    public record GuestSession(String accessToken, boolean guest, long expiresAt) {}
    public record CatalogView(int version, List<CourseView> courses) {}
    public record CourseView(String id, String title, String description, int order, boolean unlocked,
                             double unlockRatio, double completion, int learnedCount, int masteredCount,
                             List<SignView> signs) {}
    public record SignView(String signId, String displayName, String tip, int order, String state,
                           boolean unlocked, boolean learned, boolean mastered, boolean cameraVerified,
                           int quizBestScore, String sampleVideoUrl) {}
    public record DictionarySignView(String signId, String displayName, String courseId, String tip,
                                     String sampleVideoUrl, boolean learned, boolean mastered, boolean cameraVerified) {}
    public record SignListView(int total, List<DictionarySignView> items) {}
    public record ProgressSummary(int learnedCount, int masteredCount, int totalXp, int currentStreak,
                                  int progressItemCount, List<RecentActivity> recentActivity) {}
    public record RecentActivity(String signId, String status, boolean verified, String createdAt) {}
    public record QuizProgressRequest(String signId, int correct, int total) {}
    public static class GuestImportRequest { public List<GuestProgressItem> items = new ArrayList<>(); }
    public static class GuestProgressItem {
        public String signId; public boolean learned; public boolean cameraVerified;
        public int quizBestScore; public int xp;
    }
    public record DailyChallengeView(String date, int xpReward, boolean completed, List<DictionarySignView> signs) {}
    public record LeaderboardEntry(int rank, String name, String organization, int xp, boolean currentUser) {}
    public record LeaderboardView(String scope, String courseId, int total, List<LeaderboardEntry> players) {}
}
