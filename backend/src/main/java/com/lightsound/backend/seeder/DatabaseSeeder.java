package com.lightsound.backend.seeder;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.lightsound.backend.model.*;
import com.lightsound.backend.repository.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import java.io.File;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.util.*;
import org.springframework.security.crypto.bcrypt.BCrypt;

@Component
public class DatabaseSeeder implements CommandLineRunner {

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

    @Override
    @Transactional
    public void run(String... args) throws Exception {
        if (courseRepository.count() > 0) {
            System.out.println("Database already seeded.");
            return;
        }

        System.out.println("Seeding database...");

        ObjectMapper mapper = new ObjectMapper();
        ClassPathResource bundledSeed = new ClassPathResource("offline_game_data.json");
        JsonNode root;
        if (bundledSeed.exists()) {
            System.out.println("Reading bundled seed file: offline_game_data.json");
            try (var input = bundledSeed.getInputStream()) {
                root = mapper.readTree(input);
            }
        } else {
            // Backward-compatible fallback for older Unity-based checkouts.
        String[] potentialPaths = {
            "../Light_sound_exe101_G5/Assets/_Project/Resources/Offline/offline_game_data.json",
            "./Light_sound_exe101_G5/Assets/_Project/Resources/Offline/offline_game_data.json",
            "Light_sound_exe101_G5/Assets/_Project/Resources/Offline/offline_game_data.json"
        };

        File jsonFile = null;
        for (String path : potentialPaths) {
            File f = new File(path);
            if (f.exists()) {
                jsonFile = f;
                break;
            }
        }

        if (jsonFile == null) {
            // Try absolute resolution based on user directory
            String userDir = System.getProperty("user.dir");
            jsonFile = Paths.get(userDir, "Light_sound_exe101_G5", "Assets", "_Project", "Resources", "Offline", "offline_game_data.json").toFile();
            if (!jsonFile.exists()) {
                // Try parent resolution if run inside backend directory
                jsonFile = Paths.get(userDir, "..", "Light_sound_exe101_G5", "Assets", "_Project", "Resources", "Offline", "offline_game_data.json").toFile();
            }
        }

        if (!jsonFile.exists()) {
                System.err.println("offline_game_data.json not found; legacy game seed skipped.");
            return;
        }

        System.out.println("Reading seed file: " + jsonFile.getAbsolutePath());
            root = mapper.readTree(jsonFile);
        }

        // 1. Seed Profile / Default User
        JsonNode profileNode = root.get("profile");
        String defaultUsername = "offline_player";
        String name = "Offline Player";
        String orgId = "local";
        String orgName = "Local Academy";
        String imageUrl = "";

        if (profileNode != null) {
            defaultUsername = profileNode.path("userName").asText(defaultUsername);
            name = profileNode.path("name").asText(name);
            orgId = profileNode.path("orgId").asText(orgId);
            orgName = profileNode.path("orgName").asText(orgName);
            imageUrl = profileNode.path("imageUrl").asText(imageUrl);
        }

        User defaultUser = userRepository.findByUsername(defaultUsername).orElse(null);
        if (defaultUser == null) {
            defaultUser = User.builder()
                    .username(defaultUsername)
                    .passwordHash(BCrypt.hashpw("password123", BCrypt.gensalt()))
                    .name(name)
                    .orgId(orgId)
                    .orgName(orgName)
                    .imageUrl(imageUrl)
                    .build();
            defaultUser = userRepository.save(defaultUser);
        }

        // 2. Seed Courses, Lessons, Questions
        JsonNode coursesNode = root.get("courses");
        if (coursesNode != null && coursesNode.isArray()) {
            for (JsonNode courseNode : coursesNode) {
                String courseId = courseNode.get("id").asText();

                Course course = courseRepository.findById(courseId).orElse(null);
                if (course == null) {
                    course = Course.builder()
                            .id(courseId)
                            .name(courseNode.get("name").asText())
                            .description(courseNode.path("description").asText())
                            .storeDescription(courseNode.path("storeDescription").asText())
                            .price(courseNode.path("price").asDouble(0.0))
                            .startDate(LocalDateTime.now())
                            .endDate(LocalDateTime.now().plusYears(1))
                            .build();
                    course = courseRepository.save(course);
                }

                // Unlock Shapes course by default for the default user
                if ("course-shapes-01".equals(courseId)) {
                    if (userPurchaseRepository.findByUserIdAndCourseId(defaultUser.getId(), courseId).isEmpty()) {
                        UserPurchase purchase = UserPurchase.builder()
                                .user(defaultUser)
                                .course(course)
                                .build();
                        userPurchaseRepository.save(purchase);
                    }
                }

                // Seed Lessons
                JsonNode lessonsNode = courseNode.get("lessons");
                if (lessonsNode != null && lessonsNode.isArray()) {
                    for (JsonNode lessonNode : lessonsNode) {
                        String lessonId = lessonNode.get("id").asText();

                        Lesson lesson = lessonRepository.findById(lessonId).orElse(null);
                        if (lesson == null) {
                            lesson = Lesson.builder()
                                    .id(lessonId)
                                    .course(course)
                                    .title(lessonNode.get("title").asText())
                                    .orderNum(lessonNode.path("order").asInt(1))
                                    .description(lessonNode.path("description").asText())
                                    .build();
                            lesson = lessonRepository.save(lesson);
                        }

                        // Seed Questions
                        JsonNode questionsNode = lessonNode.get("questions");
                        if (questionsNode != null && questionsNode.isArray()) {
                            for (JsonNode qNode : questionsNode) {
                                String qId = qNode.get("id").asText();

                                Question question = questionRepository.findById(qId).orElse(null);
                                if (question == null) {
                                    List<String> answers = new ArrayList<>();
                                    JsonNode answersNode = qNode.get("answers");
                                    if (answersNode != null && answersNode.isArray()) {
                                        for (JsonNode ans : answersNode) {
                                            answers.add(ans.asText());
                                        }
                                    }

                                    List<Integer> correctAnswers = new ArrayList<>();
                                    JsonNode correctNode = qNode.get("correctAnswers");
                                    if (correctNode != null && correctNode.isArray()) {
                                        for (JsonNode idx : correctNode) {
                                            correctAnswers.add(idx.asInt());
                                        }
                                    }

                                    question = Question.builder()
                                            .id(qId)
                                            .lesson(lesson)
                                            .questionIndex(qNode.path("questionIndex").asInt(1))
                                            .typeId(qNode.path("type").asInt(0))
                                            .content(qNode.get("content").asText())
                                            .level(qNode.path("level").asInt(1))
                                            .answers(answers)
                                            .correctAnswers(correctAnswers)
                                            .point(qNode.path("point").asInt(100))
                                            .isHtmlContent(qNode.path("isHtmlContent").asBoolean(false))
                                            .timePerQuestion(qNode.path("timePerQuestion").asInt(0))
                                            .videoUrl(qNode.path("videoUrl").isMissingNode() ? null : qNode.path("videoUrl").asText(null))
                                            .build();
                                    questionRepository.save(question);
                                }
                            }
                        }
                    }
                }
            }
        }

        // 3. Seed Leaderboard Entries
        JsonNode leaderboardNode = root.get("courseLeaderboards");
        if (leaderboardNode != null && leaderboardNode.isArray()) {
            Map<String, User> mockUsers = new HashMap<>();
            for (JsonNode entry : leaderboardNode) {
                String pName = entry.get("name").asText();
                String pOrg = entry.get("orgName").asText();
                String cId = entry.get("courseId").asText();
                int score = entry.get("totalPoint").asInt(0);

                String mockUsername = "user_" + pName.toLowerCase();
                User mockUser = mockUsers.get(mockUsername);
                if (mockUser == null) {
                    mockUser = userRepository.findByUsername(mockUsername).orElse(null);
                    if (mockUser == null) {
                        mockUser = User.builder()
                                .username(mockUsername)
                                .passwordHash(BCrypt.hashpw("password123", BCrypt.gensalt()))
                                .name(pName)
                                .orgId("org_" + pOrg.toLowerCase().replaceAll("\\s+", "_"))
                                .orgName(pOrg)
                                .build();
                        mockUser = userRepository.save(mockUser);
                    }
                    mockUsers.put(mockUsername, mockUser);
                }

                // Unlock course package
                Course course = courseRepository.findById(cId).orElse(null);
                if (course != null) {
                    if (userPurchaseRepository.findByUserIdAndCourseId(mockUser.getId(), cId).isEmpty()) {
                        UserPurchase purchase = UserPurchase.builder()
                                .user(mockUser)
                                .course(course)
                                .build();
                        userPurchaseRepository.save(purchase);
                    }

                    // Seed attempt record for maximum score calculation
                    Lesson firstLesson = lessonRepository.findByCourseIdOrderByOrderNumAsc(cId).stream().findFirst().orElse(null);
                    if (firstLesson != null) {
                        String mockSessionId = "session_mock_" + mockUser.getUsername() + "_" + firstLesson.getId();
                        if (attemptRecordRepository.findById(mockSessionId).isEmpty()) {
                            List<Map<String, Object>> answeredQ = new ArrayList<>();
                            // Mock 1 answered question
                            Map<String, Object> aq = new HashMap<>();
                            aq.put("question_id", firstLesson.getId() + "-q1");
                            aq.put("is_correct", true);
                            aq.put("score", score);
                            answeredQ.add(aq);

                            AttemptRecord attempt = AttemptRecord.builder()
                                    .sessionId(mockSessionId)
                                    .user(mockUser)
                                    .courseId(cId)
                                    .lesson(firstLesson)
                                    .score(score)
                                    .totalCorrectAnswer(score / 100)
                                    .completionTime(45)
                                    .completedAt(LocalDateTime.now().minusDays(1))
                                    .answeredQuestions(answeredQ)
                                    .isCompleted(true)
                                    .build();
                            attemptRecordRepository.save(attempt);
                        }
                    }
                }
            }
        }

        System.out.println("Database seeded successfully.");
    }
}
