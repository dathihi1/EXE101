package com.lightsound.backend.repository;

import com.lightsound.backend.model.UserReward;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public interface UserRewardRepository extends JpaRepository<UserReward, String> {
    List<UserReward> findByUserId(String userId);
}
