package com.lightsound.backend.model;

import jakarta.persistence.*;
import lombok.*;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.UUID;

@Entity
@Table(name = "daily_challenge_completions", uniqueConstraints =
        @UniqueConstraint(name = "uk_daily_challenge_user_date", columnNames = {"user_id", "challenge_date"}))
@Getter @Setter @NoArgsConstructor @AllArgsConstructor @Builder
public class DailyChallengeCompletion {
    @Id @Builder.Default @Column(length = 36)
    private String id = UUID.randomUUID().toString();
    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;
    @Column(name = "challenge_date", nullable = false)
    private LocalDate challengeDate;
    @Column(name = "xp_awarded", nullable = false)
    @Builder.Default private int xpAwarded = 0;
    @Column(name = "completed_at", nullable = false)
    @Builder.Default private LocalDateTime completedAt = LocalDateTime.now();
}
