package com.lightsound.backend.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.util.List;

public class GameDto {

    // ----------------- Authentication / Profile -----------------
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class RewardsUserData {
        public String contestId;
        public int rank;
        public int prize;
        public String receivedDate;
        public String contestName;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class LoginRequest {
        public String username;
        public String password;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class RegisterRequest {
        public String username;
        public String password;
        public String name;
        public String orgId;
        public String orgName;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class LoginResponse {
        public String userName;
        public String imageUrl;
        public String name;
        public List<RewardsUserData> rewards;
        public String orgId;
        public String orgName;
        public String accessToken;
        public boolean guest;
    }

    // ----------------- Courses & Contests -----------------
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class ContestData {
        public String id;
        public String name;
        public String startDate;
        public String endDate;
        public String description;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class ContestResponse {
        public int total;
        public List<ContestData> items;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class TurnResponse {
        public int turns;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class RewardData {
        public int rank;
        public int price;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class ContestRewardResponse {
        public int total;
        public List<RewardData> items;
    }

    // ----------------- Gameplay / Sessions -----------------
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ContestStart {
        public String contestId;
        public String userName;
        public String orgId;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class ContestStartResponse {
        public String sessionId;
        public int turns;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class QuestionResponse {
        public int timePerQuestion;
        public int questionIndex;
        public int type;
        public String content;
        public int totalScore;
        public int level;
        public List<String> answers;
        public String id;
        public int highestScore;
        public boolean isHtmlContent;
        public String videoUrl;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class SubmitData {
        public String contestId;
        public String sessionId;
        public String questionId;
        public List<Integer> answer;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class SubmitResult {
        public int playTimeSecond;
        public boolean result;
        public List<Integer> correctAnswers;
        public int point;
        public int totalCorrectAnswer;
        public int totalScore;
        public int streakCount;
        public int bonusPoint;
        public boolean isLessonCompleted;
        public String courseId;
        public String lessonId;
    }

    // ----------------- Leaderboard -----------------
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class PlayerLeaderboard {
        public String name;
        public String orgName;
        public int rank;
        public int totalPoint;
        public int totalTime;
        public int reward;
        @Builder.Default
        public String descriptionEng = "";
        @Builder.Default
        public String descriptionMy = "";
        @Builder.Default
        public String subDescriptionEng = "";
        @Builder.Default
        public String subDescriptionMy = "";
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class TopLeaderboardResult {
        public int total;
        public List<PlayerLeaderboard> players;
    }

    // ----------------- Store -----------------
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class PackageData {
        public String id;
        public int turns;
        public int lessonCount;
        public double price;
        public String displayName;
        public String description;
        public String storeDescription;
        public boolean purchased;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class PackageResponse {
        public int total;
        public List<PackageData> items;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class BuyPackageRequest {
        public String packageId;
        public String userName;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class BuyPackageResponse {
        public int turns;
    }

    // ----------------- History & System -----------------
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class DailyHistoryResponse {
        public int dayIndex;
        public int highhestScore; // note: typo in client model 'highhestScore'
        public int completionTime;
        public int totalAttempt;
        public String datePlayed;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class AttemptData {
        public int attemptIndex;
        public int score;
        public int completionTime;
        public String datePlayed;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class DailyAttemptDetailResponse {
        public int total;
        public List<AttemptData> data;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class DateTimeNowResponse {
        public String time;
    }
}
