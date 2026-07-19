package com.lightsound.backend.repository;
import com.lightsound.backend.model.LearningProgress;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;
import java.util.Optional;
public interface LearningProgressRepository extends JpaRepository<LearningProgress, String> {
    List<LearningProgress> findByUserId(String userId);
    Optional<LearningProgress> findByUserIdAndSignId(String userId, String signId);
}
