package com.lightsound.backend.repository;

import com.lightsound.backend.model.Lesson;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public interface LessonRepository extends JpaRepository<Lesson, String> {
    List<Lesson> findByCourseIdOrderByOrderNumAsc(String courseId);
}
