package com.lightsound.backend.model;

import jakarta.persistence.*;
import lombok.*;

import java.time.LocalDateTime;
import java.util.UUID;

@Entity
@Table(name = "learning_progress", uniqueConstraints =
        @UniqueConstraint(name = "uk_learning_progress_user_sign", columnNames = {"user_id", "sign_id"}))
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class LearningProgress {
    @Id
    @Builder.Default
    @Column(length = 36)
    private String id = UUID.randomUUID().toString();

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Column(name = "course_id", nullable = false)
    private String courseId;

    @Column(name = "sign_id", nullable = false)
    private String signId;

    @Builder.Default
    private boolean learned = false;

    @Builder.Default
    private boolean mastered = false;

    @Column(name = "camera_verified")
    @Builder.Default
    private boolean cameraVerified = false;

    @Column(name = "quiz_best_score")
    @Builder.Default
    private int quizBestScore = 0;

    @Builder.Default
    private int xp = 0;

    @Column(name = "updated_at", nullable = false)
    @Builder.Default
    private LocalDateTime updatedAt = LocalDateTime.now();
}
