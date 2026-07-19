package com.lightsound.backend.model;

import jakarta.persistence.*;
import lombok.*;
import java.time.LocalDateTime;
import java.util.UUID;

@Entity
@Table(name = "practice_attempts")
@Getter @Setter @NoArgsConstructor @AllArgsConstructor @Builder
public class PracticeAttempt {
    @Id @Builder.Default @Column(length = 36)
    private String id = UUID.randomUUID().toString();
    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;
    @Column(name = "course_id", nullable = false)
    private String courseId;
    @Column(name = "sign_id", nullable = false)
    private String signId;
    @Column(name = "predicted_sign_id")
    private String predictedSignId;
    @Column(nullable = false)
    private String status;
    @Builder.Default private double confidence = 0.0;
    @Builder.Default private boolean verified = false;
    @Column(name = "created_at", nullable = false)
    @Builder.Default private LocalDateTime createdAt = LocalDateTime.now();
}
