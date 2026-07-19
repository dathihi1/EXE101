package com.lightsound.backend.model;

import jakarta.persistence.*;
import lombok.*;
import java.util.ArrayList;
import java.util.List;

@Entity
@Table(name = "questions")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Question {

    @Id
    private String id; // e.g. "course-shapes-01-lesson-01-q1"

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "lesson_id", nullable = false)
    private Lesson lesson;

    @Column(name = "question_index")
    @Builder.Default
    private Integer questionIndex = 1;

    @Column(name = "type_id")
    @Builder.Default
    private Integer typeId = 0;

    @Column(nullable = false, columnDefinition = "TEXT")
    private String content;

    @Builder.Default
    private Integer level = 1;

    @Convert(converter = StringListConverter.class)
    @Column(columnDefinition = "TEXT", nullable = false)
    @Builder.Default
    private List<String> answers = new ArrayList<>();

    @Convert(converter = IntegerListConverter.class)
    @Column(name = "correct_answers", columnDefinition = "TEXT", nullable = false)
    @Builder.Default
    private List<Integer> correctAnswers = new ArrayList<>();

    @Builder.Default
    private Integer point = 100;

    @Column(name = "is_html_content")
    @Builder.Default
    private Boolean isHtmlContent = false;

    @Column(name = "time_per_question")
    @Builder.Default
    private Integer timePerQuestion = 0;

    @Column(name = "video_url")
    private String videoUrl;
}
