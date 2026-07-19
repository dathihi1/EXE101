package com.lightsound.backend.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestClient;
import org.springframework.web.multipart.MultipartFile;
import java.util.*;

@Service
public class RecognitionGatewayService {
    public static final int MAX_FRAMES = 32;
    public static final long MAX_TOTAL_BYTES = 2L * 1024L * 1024L;
    private final RestClient client;
    private final String url;
    public RecognitionGatewayService(RestClient.Builder builder,
            @Value("${vsl.recognition.inference-url:http://localhost:7860/api/infer/frames}") String url) {
        this.client = builder.build(); this.url = url;
    }
    public Map<String, Object> recognize(String expectedSignId, List<MultipartFile> frames) {
        validate(frames);
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("expected_sign_id", expectedSignId);
        try {
            for (int i = 0; i < frames.size(); i++) body.add("frames", new NamedResource(frames.get(i).getBytes(), "frame-" + i + ".jpg"));
            @SuppressWarnings("unchecked") Map<String, Object> response = client.post().uri(url)
                    .contentType(MediaType.MULTIPART_FORM_DATA).body(body).retrieve().body(Map.class);
            return response == null ? unavailable("Dịch vụ nhận diện không trả dữ liệu.") : response;
        } catch (Exception exception) { return unavailable("Không thể kết nối dịch vụ nhận diện."); }
    }
    private void validate(List<MultipartFile> frames) {
        if (frames == null || frames.size() < 8) throw new IllegalArgumentException("Cần ít nhất 8 frame để nhận diện.");
        if (frames.size() > MAX_FRAMES) throw new IllegalArgumentException("Số frame vượt quá giới hạn 32.");
        long total = 0;
        for (MultipartFile frame : frames) {
            if (frame == null || frame.isEmpty()) throw new IllegalArgumentException("Frame không hợp lệ.");
            total += frame.getSize();
            if (frame.getContentType() != null && !MediaType.IMAGE_JPEG_VALUE.equalsIgnoreCase(frame.getContentType()))
                throw new IllegalArgumentException("Chỉ chấp nhận frame JPEG.");
        }
        if (total > MAX_TOTAL_BYTES) throw new IllegalArgumentException("Tổng dữ liệu camera vượt quá 2 MB.");
    }
    private Map<String, Object> unavailable(String hint) {
        Map<String, Object> value = new LinkedHashMap<>();
        value.put("status", "service_unavailable"); value.put("verified", false); value.put("predictedSignId", "");
        value.put("predictedLabel", ""); value.put("confidence", 0.0); value.put("top3", List.of()); value.put("qualityHints", List.of(hint));
        return value;
    }
    private static final class NamedResource extends ByteArrayResource {
        private final String filename;
        private NamedResource(byte[] bytes, String filename) { super(bytes); this.filename = filename; }
        @Override public String getFilename() { return filename; }
    }
}
