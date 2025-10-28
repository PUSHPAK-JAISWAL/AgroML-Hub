package com.AgroHub_ML.backend.service;

import com.AgroHub_ML.backend.Entity.PredictionHistory;
import com.AgroHub_ML.backend.Repositories.PredictionRepository;
import lombok.RequiredArgsConstructor;
import org.bson.types.ObjectId;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.time.Instant;
import java.util.List;
import java.util.Map;

@Service
@RequiredArgsConstructor
public class PredictionService {

    @Autowired
    private RestTemplate restTemplate;

    @Autowired
    private PredictionRepository predictionRepository;

    @Autowired
    private EmailService emailService;

    @Value("${app.fastapi.base-url}")
    private String fastApiBase;

    public Map<String,Object> predictAndSave(ObjectId userId, String endpoint, Map<String,Object> payload, String userEmail) {
        String suffix = endpoint.startsWith("/")?endpoint:"/"+endpoint;
        String url = fastApiBase+suffix;

        var respEntity = restTemplate.postForEntity(url,payload,Map.class);
        Map<String,Object> body = respEntity.getBody();

        PredictionHistory ph = PredictionHistory.builder()
                .userId(userId)
                .endpoint(suffix)
                .input(payload)
                .response(body)
                .createdAt(Instant.now())
                .build();

        predictionRepository.save(ph);

        try {
            String subject = "Your prediction result is ready";
            // Use the new HTML report sender (expects input map + results map)
            emailService.sendPredictionReport(userEmail, subject, payload, body);
        } catch (Exception ex) {
            // keep behavior: swallow/send error (don't fail prediction). optionally log.
        }
        return body;
    }

    public List<PredictionHistory> getHistoryForUser(ObjectId userId) {
        return predictionRepository.findByUserIdOrderByCreatedAtDesc(userId);
    }

}
