package com.lightsound.backend.controller;

import com.lightsound.backend.dto.GameDto.*;
import com.lightsound.backend.model.*;
import com.lightsound.backend.repository.*;
import com.lightsound.backend.service.ApiTokenService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.stream.Collectors;
import org.springframework.security.crypto.bcrypt.BCrypt;

@RestController
@RequestMapping("/api/v1")
@CrossOrigin(origins = "*")
public class GameController {

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private CourseRepository courseRepository;

    @Autowired
    private LessonRepository lessonRepository;

    @Autowired
    private QuestionRepository questionRepository;

    @Autowired
    private UserPurchaseRepository userPurchaseRepository;

    @Autowired
    private AttemptRecordRepository attemptRecordRepository;

    @Autowired
    private UserRewardRepository userRewardRepository;

    @Autowired
    private ApiTokenService apiTokenService;

    private static final DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    private String formatDateTime(LocalDateTime dt) {
        if (dt == null) return "";
        return dt.format(formatter);
    }

    // ----------------- Auth / Login Endpoints -----------------

    @PostMapping("/auth/register")
    public ResponseEntity<?> register(@RequestBody RegisterRequest req) {
        if (req.getUsername() == null || req.getUsername().trim().isEmpty() ||
            req.getPassword() == null || req.getPassword().trim().isEmpty()) {
            return ResponseEntity.badRequest().body(Map.of("message", "Tên đăng nhập và mật khẩu không được để trống."));
        }

        String usernameClean = req.getUsername().trim();
        if (userRepository.findByUsername(usernameClean).isPresent()) {
            return ResponseEntity.badRequest().body(Map.of("message", "Tên đăng nhập đã tồn tại."));
        }

        String passwordHash = BCrypt.hashpw(req.getPassword(), BCrypt.gensalt());

        User user = User.builder()
                .username(usernameClean)
                .passwordHash(passwordHash)
                .name(req.getName() != null && !req.getName().trim().isEmpty() ? req.getName().trim() : usernameClean)
                .orgId(req.getOrgId() != null ? req.getOrgId() : "local")
                .orgName(req.getOrgName() != null ? req.getOrgName() : "Local Academy")
                .build();
        user = userRepository.save(user);

        // Auto unlock basic shape course for new user (simulates database seeder behavior for shapes)
        Course shapesCourse = courseRepository.findById("course-shapes-01").orElse(null);
        if (shapesCourse != null) {
            UserPurchase purchase = UserPurchase.builder()
                    .user(user)
                    .course(shapesCourse)
                    .build();
            userPurchaseRepository.save(purchase);
        }

        return ResponseEntity.ok(LoginResponse.builder()
                .userName(user.getUsername())
                .imageUrl(user.getImageUrl() != null ? user.getImageUrl() : "")
                .name(user.getName())
                .rewards(Collections.emptyList())
                .orgId(user.getOrgId() != null ? user.getOrgId() : "")
                .orgName(user.getOrgName() != null ? user.getOrgName() : "")
                .accessToken(apiTokenService.issueUserToken(user.getId()))
                .guest(false)
                .build());
    }

    @PostMapping("/auth/login")
    public ResponseEntity<?> login(@RequestBody LoginRequest req) {
        if (req.getUsername() == null || req.getUsername().trim().isEmpty() ||
            req.getPassword() == null || req.getPassword().trim().isEmpty()) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(Map.of("message", "Tên đăng nhập và mật khẩu không được để trống."));
        }

        String usernameClean = req.getUsername().trim();
        User user = userRepository.findByUsername(usernameClean).orElse(null);
        if (user == null || user.getPasswordHash() == null || user.getPasswordHash().isEmpty() ||
            !BCrypt.checkpw(req.getPassword(), user.getPasswordHash())) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(Map.of("message", "Tên đăng nhập hoặc mật khẩu không chính xác."));
        }

        List<RewardsUserData> rewards = userRewardRepository.findByUserId(user.getId()).stream()
                .map(r -> RewardsUserData.builder()
                        .contestId(r.getCourse().getId())
                        .rank(r.getRank())
                        .prize(r.getPrize())
                        .receivedDate(formatDateTime(r.getReceivedDate()))
                        .contestName(r.getCourse().getName())
                        .build())
                .collect(Collectors.toList());

        return ResponseEntity.ok(LoginResponse.builder()
                .userName(user.getUsername())
                .imageUrl(user.getImageUrl() != null ? user.getImageUrl() : "")
                .name(user.getName())
                .rewards(rewards)
                .orgId(user.getOrgId() != null ? user.getOrgId() : "")
                .orgName(user.getOrgName() != null ? user.getOrgName() : "")
                .accessToken(apiTokenService.issueUserToken(user.getId()))
                .guest(false)
                .build());
    }

    // ----------------- Courses & Contests Endpoints -----------------

    @GetMapping("/courses")
    public ResponseEntity<ContestResponse> getCourses() {
        List<Course> courses = courseRepository.findAll();
        List<ContestData> items = courses.stream()
                .map(c -> ContestData.builder()
                        .id(c.getId())
                        .name(c.getName())
                        .startDate(formatDateTime(c.getStartDate()))
                        .endDate(formatDateTime(c.getEndDate()))
                        .description(c.getDescription() != null ? c.getDescription() : "")
                        .build())
                .collect(Collectors.toList());

        return ResponseEntity.ok(ContestResponse.builder()
                .total(items.size())
                .items(items)
                .build());
    }

    @GetMapping("/courses/{courseId}/turns")
    public ResponseEntity<TurnResponse> getCourseTurns(@PathVariable String courseId) {
        Course course = courseRepository.findById(courseId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Course not found"));

        int lessonCount = lessonRepository.findByCourseIdOrderByOrderNumAsc(courseId).size();
        return ResponseEntity.ok(TurnResponse.builder()
                .turns(Math.max(lessonCount, 1))
                .build());
    }

    @GetMapping("/courses/{courseId}/rewards")
    public ResponseEntity<ContestRewardResponse> getCourseRewards(@PathVariable String courseId) {
        List<RewardData> rewards = List.of(
                new RewardData(1, 1000),
                new RewardData(2, 500),
                new RewardData(3, 250)
        );
        return ResponseEntity.ok(ContestRewardResponse.builder()
                .total(rewards.size())
                .items(rewards)
                .build());
    }

    // ----------------- Gameplay Session Endpoints -----------------

    @PostMapping("/sessions/start")
    public ResponseEntity<ContestStartResponse> startSession(
            @RequestBody ContestStart req,
            @RequestParam(defaultValue = "0") int lessonIndex) {

        User user = userRepository.findByUsername(req.getUserName()).orElse(null);
        if (user == null) {
            user = User.builder()
                    .username(req.getUserName())
                    .name(req.getUserName())
                    .orgId(req.getOrgId())
                    .orgName("Local Academy")
                    .build();
            user = userRepository.save(user);
        }

        Course course = courseRepository.findById(req.getContestId())
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Course not found"));

        List<Lesson> lessons = lessonRepository.findByCourseIdOrderByOrderNumAsc(req.getContestId());
        if (lessons.isEmpty()) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "No lessons found for this course");
        }

        int idx = Math.max(0, Math.min(lessonIndex, lessons.size() - 1));
        Lesson lesson = lessons.get(idx);

        String sessionId = UUID.randomUUID().toString().replace("-", "");

        AttemptRecord attempt = AttemptRecord.builder()
                .sessionId(sessionId)
                .user(user)
                .courseId(course.getId())
                .lesson(lesson)
                .score(0)
                .totalCorrectAnswer(0)
                .completionTime(0)
                .answeredQuestions(new ArrayList<>())
                .isCompleted(false)
                .completedAt(LocalDateTime.now())
                .build();
        attemptRecordRepository.save(attempt);

        return ResponseEntity.ok(ContestStartResponse.builder()
                .sessionId(sessionId)
                .turns(lessons.size())
                .build());
    }

    @GetMapping("/sessions/{sessionId}/questions/next")
    public ResponseEntity<QuestionResponse> getNextQuestion(@PathVariable String sessionId) {
        AttemptRecord attempt = attemptRecordRepository.findById(sessionId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Session not found"));

        if (attempt.getIsCompleted()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Lesson session already completed");
        }

        List<Question> questions = questionRepository.findByLessonIdOrderByQuestionIndexAsc(attempt.getLesson().getId());
        if (questions.isEmpty()) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "No questions found for this lesson");
        }

        // Determine answered question IDs
        List<Map<String, Object>> answeredList = attempt.getAnsweredQuestions();
        Set<String> answeredIds = answeredList == null ? new HashSet<>() : answeredList.stream()
                .map(aq -> (String) aq.get("question_id"))
                .collect(Collectors.toSet());

        // Find the first unanswered question
        Question nextQ = null;
        for (Question q : questions) {
            if (!answeredIds.contains(q.getId())) {
                nextQ = q;
                break;
            }
        }

        if (nextQ == null) {
            attempt.setIsCompleted(true);
            attemptRecordRepository.save(attempt);
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "All questions answered");
        }

        // Find highest score in previous COMPLETED attempts
        int highestScore = attemptRecordRepository.findByUserIdAndCourseIdAndIsCompletedTrue(attempt.getUser().getId(), attempt.getCourseId()).stream()
                .filter(pa -> pa.getLesson().getId().equals(attempt.getLesson().getId()))
                .filter(pa -> !pa.getSessionId().equals(sessionId))
                .mapToInt(AttemptRecord::getScore)
                .max()
                .orElse(0);

        return ResponseEntity.ok(QuestionResponse.builder()
                .timePerQuestion(nextQ.getTimePerQuestion())
                .questionIndex(answeredIds.size() + 1)
                .type(nextQ.getTypeId())
                .content(nextQ.getContent())
                .totalScore(attempt.getScore())
                .level(nextQ.getLevel())
                .answers(nextQ.getAnswers())
                .id(nextQ.getId())
                .highestScore(highestScore)
                .isHtmlContent(nextQ.getIsHtmlContent())
                .videoUrl(nextQ.getVideoUrl())
                .build());
    }

    @PostMapping("/sessions/{sessionId}/submit")
    public ResponseEntity<SubmitResult> submitAnswer(
            @PathVariable String sessionId,
            @RequestBody SubmitData req) {

        AttemptRecord attempt = attemptRecordRepository.findById(sessionId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Session not found"));

        if (attempt.getIsCompleted()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Session already completed");
        }

        Question question = questionRepository.findById(req.getQuestionId())
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Question not found"));

        List<Map<String, Object>> answeredList = attempt.getAnsweredQuestions();
        if (answeredList == null) {
            answeredList = new ArrayList<>();
        }

        boolean alreadyAnswered = answeredList.stream()
                .anyMatch(aq -> question.getId().equals(aq.get("question_id")));
        if (alreadyAnswered) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Question already answered in this session");
        }

        // Validate answer
        List<Integer> submittedAnswer = req.getAnswer() != null ? req.getAnswer() : new ArrayList<>();
        List<Integer> correctAnswers = question.getCorrectAnswers() != null ? question.getCorrectAnswers() : new ArrayList<>();

        List<Integer> sortedSubmitted = new ArrayList<>(submittedAnswer);
        List<Integer> sortedCorrect = new ArrayList<>(correctAnswers);
        Collections.sort(sortedSubmitted);
        Collections.sort(sortedCorrect);

        boolean isCorrect = sortedSubmitted.equals(sortedCorrect);
        int awardedPoints = isCorrect ? question.getPoint() : 0;

        // Calculate streak
        int streak = 0;
        if (isCorrect) {
            streak = 1;
            for (int i = answeredList.size() - 1; i >= 0; i--) {
                Boolean wasCorrect = (Boolean) answeredList.get(i).get("is_correct");
                if (Boolean.TRUE.equals(wasCorrect)) {
                    streak++;
                } else {
                    break;
                }
            }
        }

        int bonus = 0;
        if (isCorrect && streak >= 3) {
            bonus = 10 * streak;
            awardedPoints += bonus;
        }

        // Save progress
        Map<String, Object> progress = new HashMap<>();
        progress.put("question_id", question.getId());
        progress.put("is_correct", isCorrect);
        progress.put("score", awardedPoints);
        answeredList.add(progress);

        attempt.setAnsweredQuestions(answeredList);
        attempt.setScore(attempt.getScore() + awardedPoints);
        if (isCorrect) {
            attempt.setTotalCorrectAnswer(attempt.getTotalCorrectAnswer() + 1);
        }

        // Check completion
        int totalQuestions = questionRepository.findByLessonIdOrderByQuestionIndexAsc(attempt.getLesson().getId()).size();
        boolean isCompleted = answeredList.size() >= totalQuestions;
        int elapsedTime = answeredList.size() * 15;

        if (isCompleted) {
            attempt.setIsCompleted(true);
            attempt.setCompletionTime(elapsedTime);
            attempt.setCompletedAt(LocalDateTime.now());
        }

        attemptRecordRepository.save(attempt);

        return ResponseEntity.ok(SubmitResult.builder()
                .playTimeSecond(isCompleted ? attempt.getCompletionTime() : elapsedTime)
                .result(isCorrect)
                .correctAnswers(correctAnswers)
                .point(awardedPoints)
                .totalCorrectAnswer(attempt.getTotalCorrectAnswer())
                .totalScore(attempt.getScore())
                .streakCount(streak)
                .bonusPoint(bonus)
                .isLessonCompleted(attempt.getIsCompleted())
                .courseId(attempt.getCourseId())
                .lessonId(attempt.getLesson().getId())
                .build());
    }

    // ----------------- Leaderboard Endpoints -----------------

    @GetMapping("/courses/{courseId}/leaderboard")
    public ResponseEntity<TopLeaderboardResult> getCourseLeaderboard(@PathVariable String courseId) {
        List<AttemptRecord> allAttempts = attemptRecordRepository.findByCourseIdAndIsCompletedTrue(courseId);

        // Group by user and find max score, min time
        Map<String, AttemptRecord> userBestAttempts = new HashMap<>();
        for (AttemptRecord att : allAttempts) {
            String userId = att.getUser().getId();
            AttemptRecord best = userBestAttempts.get(userId);
            if (best == null || att.getScore() > best.getScore() ||
                (att.getScore().equals(best.getScore()) && att.getCompletionTime() < best.getCompletionTime())) {
                userBestAttempts.put(userId, att);
            }
        }

        List<AttemptRecord> sortedBest = new ArrayList<>(userBestAttempts.values());
        // Sort: Score DESC, Time ASC, Name ASC
        sortedBest.sort((a, b) -> {
            int scoreCompare = b.getScore().compareTo(a.getScore());
            if (scoreCompare != 0) return scoreCompare;
            int timeCompare = a.getCompletionTime().compareTo(b.getCompletionTime());
            if (timeCompare != 0) return timeCompare;
            return a.getUser().getName().compareTo(b.getUser().getName());
        });

        List<PlayerLeaderboard> players = new ArrayList<>();
        for (int i = 0; i < sortedBest.size(); i++) {
            AttemptRecord att = sortedBest.get(i);
            int rank = i + 1;
            int reward = rank == 1 ? 1000 : (rank == 2 ? 500 : (rank == 3 ? 250 : 0));
            players.add(PlayerLeaderboard.builder()
                    .name(att.getUser().getName())
                    .orgName(att.getUser().getOrgName() != null ? att.getUser().getOrgName() : "Local Academy")
                    .rank(rank)
                    .totalPoint(att.getScore())
                    .totalTime(att.getCompletionTime())
                    .reward(reward)
                    .descriptionEng("Rank " + rank + " globally")
                    .descriptionMy("Hạng " + rank + " toàn cầu")
                    .subDescriptionEng("Well played!")
                    .subDescriptionMy("Chơi rất tốt!")
                    .build());
        }

        return ResponseEntity.ok(TopLeaderboardResult.builder()
                .total(players.size())
                .players(players)
                .build());
    }

    // ----------------- Store Endpoints -----------------

    @GetMapping("/store/packages")
    public ResponseEntity<PackageResponse> getPackages(@RequestParam String userName) {
        User user = userRepository.findByUsername(userName).orElse(null);
        List<Course> courses = courseRepository.findAll();

        List<PackageData> items = new ArrayList<>();
        for (Course c : courses) {
            int lessonCount = lessonRepository.findByCourseIdOrderByOrderNumAsc(c.getId()).size();

            boolean purchased = "course-shapes-01".equals(c.getId());
            if (user != null && !purchased) {
                purchased = userPurchaseRepository.findByUserIdAndCourseId(user.getId(), c.getId()).isPresent();
            }

            items.add(PackageData.builder()
                    .id(c.getId())
                    .turns(lessonCount)
                    .lessonCount(lessonCount)
                    .price(c.getPrice())
                    .displayName(c.getName())
                    .description(c.getDescription() != null ? c.getDescription() : "")
                    .storeDescription(c.getStoreDescription() != null ? c.getStoreDescription() : "")
                    .purchased(purchased)
                    .build());
        }

        return ResponseEntity.ok(PackageResponse.builder()
                .total(items.size())
                .items(items)
                .build());
    }

    @PostMapping("/store/buy")
    public ResponseEntity<BuyPackageResponse> buyPackage(@RequestBody BuyPackageRequest req) {
        User user = userRepository.findByUsername(req.getUserName())
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "User not found"));

        Course course = courseRepository.findById(req.getPackageId())
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Course package not found"));

        if (userPurchaseRepository.findByUserIdAndCourseId(user.getId(), course.getId()).isEmpty()) {
            UserPurchase purchase = UserPurchase.builder()
                    .user(user)
                    .course(course)
                    .build();
            userPurchaseRepository.save(purchase);
        }

        int lessonCount = lessonRepository.findByCourseIdOrderByOrderNumAsc(course.getId()).size();
        return ResponseEntity.ok(BuyPackageResponse.builder()
                .turns(Math.max(lessonCount, 1))
                .build());
    }

    // ----------------- History Endpoints -----------------

    @GetMapping("/courses/{courseId}/history/daily")
    public ResponseEntity<DailyHistoryResponse> getDailyHistory(
            @PathVariable String courseId,
            @RequestParam String date,
            @RequestParam String userName) {

        User user = userRepository.findByUsername(userName)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "User not found"));

        List<AttemptRecord> attempts = attemptRecordRepository.findByUserIdAndCourseIdAndIsCompletedTrue(user.getId(), courseId);

        // Filter by date prefix: YYYY-MM-DD
        List<AttemptRecord> dailyAttempts = attempts.stream()
                .filter(att -> att.getCompletedAt().toLocalDate().toString().equals(date))
                .collect(Collectors.toList());

        int highestScore = 0;
        int completionTime = 0;

        for (AttemptRecord att : dailyAttempts) {
            if (att.getScore() >= highestScore) {
                highestScore = att.getScore();
                completionTime = att.getCompletionTime();
            }
        }

        return ResponseEntity.ok(DailyHistoryResponse.builder()
                .dayIndex(0)
                .highhestScore(highestScore)
                .completionTime(completionTime)
                .totalAttempt(dailyAttempts.size())
                .datePlayed(date + " 00:00:00")
                .build());
    }

    @GetMapping("/courses/{courseId}/history/attempts")
    public ResponseEntity<DailyAttemptDetailResponse> getDailyAttemptDetail(
            @PathVariable String courseId,
            @RequestParam String date,
            @RequestParam String userName) {

        User user = userRepository.findByUsername(userName)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "User not found"));

        List<AttemptRecord> attempts = attemptRecordRepository.findByUserIdAndCourseIdAndIsCompletedTrue(user.getId(), courseId);

        List<AttemptRecord> dailyAttempts = attempts.stream()
                .filter(att -> att.getCompletedAt().toLocalDate().toString().equals(date))
                .sorted(Comparator.comparing(AttemptRecord::getCompletedAt))
                .collect(Collectors.toList());

        List<AttemptData> dataItems = new ArrayList<>();
        for (int i = 0; i < dailyAttempts.size(); i++) {
            AttemptRecord att = dailyAttempts.get(i);
            dataItems.add(AttemptData.builder()
                    .attemptIndex(i + 1)
                    .score(att.getScore())
                    .completionTime(att.getCompletionTime())
                    .datePlayed(formatDateTime(att.getCompletedAt()))
                    .build());
        }

        return ResponseEntity.ok(DailyAttemptDetailResponse.builder()
                .total(dataItems.size())
                .data(dataItems)
                .build());
    }

    // ----------------- System Endpoints -----------------

    @GetMapping("/system/time")
    public ResponseEntity<DateTimeNowResponse> getSystemTime() {
        return ResponseEntity.ok(DateTimeNowResponse.builder()
                .time(formatDateTime(LocalDateTime.now()))
                .build());
    }
}
