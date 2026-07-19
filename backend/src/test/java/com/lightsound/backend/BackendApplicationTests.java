package com.lightsound.backend;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.lightsound.backend.dto.GameDto.*;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;
import org.springframework.mock.web.MockMultipartFile;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
class BackendApplicationTests {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Test
    void contextLoads() {
        // Simple sanity check that context boots up successfully
    }

    @Test
    void testRegisterAndLogin() throws Exception {
        // 1. Register a new user
        RegisterRequest regReq = RegisterRequest.builder()
                .username("new_test_user")
                .password("secure_pass_123")
                .name("New Test User")
                .orgId("test_org")
                .orgName("Test Academy")
                .build();

        MvcResult regResult = mockMvc.perform(post("/api/v1/auth/register")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(regReq)))
                .andExpect(status().isOk())
                .andReturn();

        LoginResponse regResp = objectMapper.readValue(regResult.getResponse().getContentAsString(), LoginResponse.class);
        assertEquals("new_test_user", regResp.getUserName());
        assertEquals("New Test User", regResp.getName());
        assertEquals("test_org", regResp.getOrgId());
        assertEquals("Test Academy", regResp.getOrgName());

        // 2. Login with correct credentials
        LoginRequest loginReq = new LoginRequest("new_test_user", "secure_pass_123");
        MvcResult loginResult = mockMvc.perform(post("/api/v1/auth/login")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(loginReq)))
                .andExpect(status().isOk())
                .andReturn();

        LoginResponse loginResp = objectMapper.readValue(loginResult.getResponse().getContentAsString(), LoginResponse.class);
        assertEquals("new_test_user", loginResp.getUserName());
        assertEquals("New Test User", loginResp.getName());

        // 3. Login with wrong credentials (should fail)
        LoginRequest wrongLoginReq = new LoginRequest("new_test_user", "wrong_pass");
        mockMvc.perform(post("/api/v1/auth/login")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(wrongLoginReq)))
                .andExpect(status().isUnauthorized());
    }

    @Test
    void testGetCourses() throws Exception {
        MvcResult result = mockMvc.perform(get("/api/v1/courses"))
                .andExpect(status().isOk())
                .andReturn();

        ContestResponse resp = objectMapper.readValue(result.getResponse().getContentAsString(), ContestResponse.class);
        assertTrue(resp.getTotal() >= 2);
        assertTrue(resp.getItems().stream().anyMatch(c -> "course-shapes-01".equals(c.getId())));
        assertTrue(resp.getItems().stream().anyMatch(c -> "course-sounds-01".equals(c.getId())));
    }

    @Test
    void testGetCourseTurns() throws Exception {
        mockMvc.perform(get("/api/v1/courses/course-shapes-01/turns"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.turns").value(4)); // Shapes Starter has 4 lessons in json
    }

    @Test
    void testSessionGameplayFlow() throws Exception {
        // 1. Start Session
        ContestStart startReq = new ContestStart("course-shapes-01", "game_player", "local");
        MvcResult startResult = mockMvc.perform(post("/api/v1/sessions/start")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(startReq))
                        .param("lessonIndex", "0"))
                .andExpect(status().isOk())
                .andReturn();

        ContestStartResponse startResp = objectMapper.readValue(startResult.getResponse().getContentAsString(), ContestStartResponse.class);
        String sessionId = startResp.getSessionId();
        assertNotNull(sessionId);

        // 2. Get Next Question
        MvcResult qResult = mockMvc.perform(get("/api/v1/sessions/" + sessionId + "/questions/next"))
                .andExpect(status().isOk())
                .andReturn();

        QuestionResponse qResp = objectMapper.readValue(qResult.getResponse().getContentAsString(), QuestionResponse.class);
        assertEquals(1, qResp.getQuestionIndex());
        assertNotNull(qResp.getId());
        String questionId = qResp.getId();

        // 3. Submit correct answer
        // From offline_game_data.json: first question correct answer index is 0
        SubmitData submitReq = new SubmitData("course-shapes-01", sessionId, questionId, List.of(0));
        MvcResult submitResult = mockMvc.perform(post("/api/v1/sessions/" + sessionId + "/submit")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(submitReq)))
                .andExpect(status().isOk())
                .andReturn();

        SubmitResult subResp = objectMapper.readValue(submitResult.getResponse().getContentAsString(), SubmitResult.class);
        assertTrue(subResp.isResult());
        assertEquals(1, subResp.getTotalCorrectAnswer());
        assertEquals(100, subResp.getTotalScore());
    }

    @Test
    void testLeaderboard() throws Exception {
        MvcResult result = mockMvc.perform(get("/api/v1/courses/course-shapes-01/leaderboard"))
                .andExpect(status().isOk())
                .andReturn();

        TopLeaderboardResult resp = objectMapper.readValue(result.getResponse().getContentAsString(), TopLeaderboardResult.class);
        assertNotNull(resp.getPlayers());
        // Mia and Leo from seeder should be ranked
        assertTrue(resp.getPlayers().size() >= 2);
        assertEquals(1, resp.getPlayers().get(0).getRank());
    }

    @Test
    void testStorePackages() throws Exception {
        mockMvc.perform(get("/api/v1/store/packages").param("userName", "game_player"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.total").value(2))
                .andExpect(jsonPath("$.items[0].id").value("course-shapes-01"))
                .andExpect(jsonPath("$.items[0].purchased").value(true)); // shapes is purchased by default
    }

    @Test
    void testSystemTime() throws Exception {
        mockMvc.perform(get("/api/v1/system/time"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.time").isNotEmpty());
    }

    @Test
    void testVslCatalogGuestAndProgressFlow() throws Exception {
        MvcResult guestResult = mockMvc.perform(post("/api/v1/auth/guest")
                        .contentType(MediaType.APPLICATION_JSON).content("{}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.guest").value(true))
                .andExpect(jsonPath("$.accessToken").isNotEmpty())
                .andReturn();
        String guestToken = objectMapper.readTree(guestResult.getResponse().getContentAsString()).get("accessToken").asText();

        mockMvc.perform(get("/api/v1/learning/catalog"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.courses.length()").value(4))
                .andExpect(jsonPath("$.courses[0].title").value("Gia đình và xưng hô"));
        mockMvc.perform(get("/api/v1/signs"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.total").value(30))
                .andExpect(jsonPath("$.items[29].displayName").value("Ướt"));

        RegisterRequest register = RegisterRequest.builder()
                .username("vsl_progress_user").password("password123")
                .name("Người học VSL").orgId("local").orgName("VSL Academy").build();
        MvcResult registerResult = mockMvc.perform(post("/api/v1/auth/register")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(register)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.accessToken").isNotEmpty())
                .andReturn();
        String userToken = objectMapper.readTree(registerResult.getResponse().getContentAsString()).get("accessToken").asText();

        mockMvc.perform(post("/api/v1/learning/progress/quiz")
                        .header("Authorization", "Bearer " + userToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"signId\":\"anh\",\"correct\":2,\"total\":3}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.learnedCount").value(1))
                .andExpect(jsonPath("$.totalXp").value(20));

        mockMvc.perform(post("/api/v1/challenges/today/complete")
                        .header("Authorization", "Bearer " + userToken))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.totalXp").value(120));
        mockMvc.perform(post("/api/v1/challenges/today/complete")
                        .header("Authorization", "Bearer " + userToken))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.totalXp").value(120));

        MockMultipartFile oneFrame = new MockMultipartFile("frames", "frame.jpg", "image/jpeg", new byte[] {1, 2, 3});
        mockMvc.perform(multipart("/api/v1/practice/recognize")
                        .file(oneFrame).param("expectedSignId", "anh").param("lessonId", "anh")
                        .header("Authorization", "Bearer " + guestToken))
                .andExpect(status().isBadRequest());
    }

    @Test
    void testGuestImportIsIdempotent() throws Exception {
        RegisterRequest register = RegisterRequest.builder()
                .username("vsl_import_user").password("password123").name("Import User").build();
        MvcResult result = mockMvc.perform(post("/api/v1/auth/register")
                        .contentType(MediaType.APPLICATION_JSON).content(objectMapper.writeValueAsString(register)))
                .andExpect(status().isOk()).andReturn();
        JsonNode response = objectMapper.readTree(result.getResponse().getContentAsString());
        String token = response.get("accessToken").asText();
        String payload = "{\"items\":[{\"signId\":\"chau\",\"learned\":true,\"cameraVerified\":true,\"quizBestScore\":3,\"xp\":50}]}";
        for (int i = 0; i < 2; i++) {
            mockMvc.perform(post("/api/v1/progress/guest-import")
                            .header("Authorization", "Bearer " + token)
                            .contentType(MediaType.APPLICATION_JSON).content(payload))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.learnedCount").value(1))
                    .andExpect(jsonPath("$.masteredCount").value(1))
                    .andExpect(jsonPath("$.totalXp").value(50));
        }
    }
}
