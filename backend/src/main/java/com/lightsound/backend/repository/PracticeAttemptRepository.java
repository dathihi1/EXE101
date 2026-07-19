package com.lightsound.backend.repository;
import com.lightsound.backend.model.PracticeAttempt;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;
public interface PracticeAttemptRepository extends JpaRepository<PracticeAttempt, String> {
    List<PracticeAttempt> findTop20ByUserIdOrderByCreatedAtDesc(String userId);
}
