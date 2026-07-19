package com.lightsound.backend.model;

import jakarta.persistence.*;
import lombok.*;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

@Entity
@Table(name = "users")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class User {

    @Id
    @Column(length = 36)
    @Builder.Default
    private String id = UUID.randomUUID().toString();

    @Column(nullable = false, unique = true)
    private String username;

    @Column(nullable = false)
    private String name;

    @Column(name = "org_id")
    private String orgId;

    @Column(name = "org_name")
    private String orgName;

    @Column(name = "image_url")
    @Builder.Default
    private String imageUrl = "";

    @Column(name = "password_hash", nullable = false)
    @Builder.Default
    private String passwordHash = "";

    @Column(name = "created_at")
    @Builder.Default
    private LocalDateTime createdAt = LocalDateTime.now();

    @OneToMany(mappedBy = "user", cascade = CascadeType.ALL, orphanRemoval = true)
    @Builder.Default
    private List<UserPurchase> purchases = new ArrayList<>();

    @OneToMany(mappedBy = "user", cascade = CascadeType.ALL, orphanRemoval = true)
    @Builder.Default
    private List<AttemptRecord> attempts = new ArrayList<>();

    @OneToMany(mappedBy = "user", cascade = CascadeType.ALL, orphanRemoval = true)
    @Builder.Default
    private List<UserReward> rewards = new ArrayList<>();
}
