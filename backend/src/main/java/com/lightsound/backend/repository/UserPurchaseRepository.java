package com.lightsound.backend.repository;

import com.lightsound.backend.model.UserPurchase;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.util.Optional;

@Repository
public interface UserPurchaseRepository extends JpaRepository<UserPurchase, String> {
    Optional<UserPurchase> findByUserIdAndCourseId(String userId, String courseId);
}
