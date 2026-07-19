package com.lightsound.backend.repository;

import com.lightsound.backend.model.Question;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public interface QuestionRepository extends JpaRepository<Question, String> {
    List<Question> findByLessonIdOrderByQuestionIndexAsc(String lessonId);
}
