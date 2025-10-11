package com.AgroHub_ML.backend.Entity;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.RequiredArgsConstructor;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

import java.time.Instant;
import java.util.Map;

@Document(collection = "prediction_history")
@AllArgsConstructor
@RequiredArgsConstructor
@NoArgsConstructor
@Data
public class PredictionHistory {

    @Id
    private String id;
    private String userId;
    private String endpoint;
    private Map<String,Object> input;
    private Map<String,Object> response;
    private Instant createdAt;

}
