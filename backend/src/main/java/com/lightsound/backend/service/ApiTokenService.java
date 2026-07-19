package com.lightsound.backend.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.Base64;
import java.util.Optional;
import java.util.UUID;

@Service
public class ApiTokenService {
    private final byte[] secret;
    public ApiTokenService(@Value("${vsl.auth.secret:change-this-secret}") String secret) {
        this.secret = secret.getBytes(StandardCharsets.UTF_8);
    }
    public String issueUserToken(String userId) { return issue("user", userId, 30L * 86400L); }
    public String issueGuestToken() { return issue("guest", UUID.randomUUID().toString(), 86400L); }
    public Optional<Session> parseBearer(String authorization) {
        if (authorization == null || !authorization.startsWith("Bearer ")) return Optional.empty();
        try {
            String decoded = new String(Base64.getUrlDecoder().decode(authorization.substring(7).trim()), StandardCharsets.UTF_8);
            String[] parts = decoded.split("\\.", 4);
            if (parts.length != 4) return Optional.empty();
            long expiresAt = Long.parseLong(parts[2]);
            String unsigned = parts[0] + "." + parts[1] + "." + parts[2];
            if (expiresAt <= Instant.now().getEpochSecond() || !MessageDigest.isEqual(parts[3].getBytes(StandardCharsets.UTF_8), sign(unsigned).getBytes(StandardCharsets.UTF_8))) return Optional.empty();
            return Optional.of(new Session(parts[0], parts[1], expiresAt));
        } catch (RuntimeException ignored) { return Optional.empty(); }
    }
    private String issue(String type, String subject, long lifetime) {
        long expires = Instant.now().plusSeconds(lifetime).getEpochSecond();
        String unsigned = type + "." + subject + "." + expires;
        return Base64.getUrlEncoder().withoutPadding().encodeToString((unsigned + "." + sign(unsigned)).getBytes(StandardCharsets.UTF_8));
    }
    private String sign(String value) {
        try {
            Mac mac = Mac.getInstance("HmacSHA256");
            mac.init(new SecretKeySpec(secret, "HmacSHA256"));
            return Base64.getUrlEncoder().withoutPadding().encodeToString(mac.doFinal(value.getBytes(StandardCharsets.UTF_8)));
        } catch (Exception exception) { throw new IllegalStateException(exception); }
    }
    public record Session(String type, String subject, long expiresAt) {
        public boolean isGuest() { return "guest".equals(type); }
        public boolean isUser() { return "user".equals(type); }
    }
}
