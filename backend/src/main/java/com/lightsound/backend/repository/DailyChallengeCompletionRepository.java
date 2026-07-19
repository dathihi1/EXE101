package com.lightsound.backend.repository;
import com.lightsound.backend.model.DailyChallengeCompletion;
import org.springframework.data.jpa.repository.JpaRepository;
import java.time.LocalDate;
import java.util.List;
import java.util.Optional;
public interface DailyChallengeCompletionRepository extends JpaRepository<DailyChallengeCompletion, String> {
    Optional<DailyChallengeCompletion> findByUserIdAndChallengeDate(String userId, LocalDate challengeDate);
    List<DailyChallengeCompletion> findByUserIdOrderByChallengeDateDesc(String userId);
}
