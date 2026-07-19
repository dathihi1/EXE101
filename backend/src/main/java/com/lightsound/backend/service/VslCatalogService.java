package com.lightsound.backend.service;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PostConstruct;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;
import java.io.InputStream;
import java.util.*;

@Service
public class VslCatalogService {
    private final ObjectMapper mapper;
    private Catalog catalog;
    private Map<String, Sign> signsById;
    public VslCatalogService(ObjectMapper mapper) { this.mapper = mapper; }
    @PostConstruct void load() {
        try (InputStream stream = new ClassPathResource("vsl_catalog.json").getInputStream()) {
            catalog = mapper.readValue(stream, Catalog.class);
        } catch (Exception exception) { throw new IllegalStateException("Cannot load vsl_catalog.json", exception); }
        catalog.courses.sort(Comparator.comparingInt(Course::getOrder));
        signsById = new LinkedHashMap<>();
        for (Course course : catalog.courses) for (int i = 0; i < course.signs.size(); i++) {
            Sign sign = course.signs.get(i); sign.courseId = course.id; sign.order = i + 1;
            if (signsById.put(sign.signId, sign) != null) throw new IllegalStateException("Duplicate signId: " + sign.signId);
        }
    }
    public List<Course> courses() { return Collections.unmodifiableList(catalog.courses); }
    public List<Sign> signs() { return List.copyOf(signsById.values()); }
    public Optional<Sign> findSign(String id) { return Optional.ofNullable(signsById.get(id)); }
    public Optional<Course> findCourse(String id) { return catalog.courses.stream().filter(c -> c.id.equals(id)).findFirst(); }

    @Getter @Setter @NoArgsConstructor @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Catalog { private int version; private List<Course> courses = new ArrayList<>(); }
    @Getter @Setter @NoArgsConstructor @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Course {
        private String id; private String title; private String description; private int order; private double unlockRatio;
        private List<Sign> signs = new ArrayList<>();
    }
    @Getter @Setter @NoArgsConstructor @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Sign {
        private String signId; private String displayName; private int sampleIndex; private String tip; private String courseId; private int order;
    }
}
