package com.lightsound.backend.repository;

import com.lightsound.backend.model.AttemptRecord;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public interface AttemptRecordRepository extends JpaRepository<AttemptRecord, String> {
    List<AttemptRecord> findByUserIdAndLessonIdAndIsCompletedTrue(String userId, String lessonId);
    List<AttemptRecord> findByCourseIdAndIsCompletedTrue(String courseId);
    List<AttemptRecord> findByUserIdAndCourseIdAndIsCompletedTrue(String userId, String courseId);
}
