package com.AgroHub_ML.backend.Entity;

import lombok.*;
import org.bson.types.ObjectId;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.stereotype.Component;

import java.time.Instant;
import java.util.Map;

@Document(collection = "prediction_history")
@AllArgsConstructor
@RequiredArgsConstructor
@Data
@Builder
public class PredictionHistory {

    @Id
    private ObjectId id;
    private ObjectId userId;
    private String endpoint;
    private Map<String,Object> input;
    private Map<String,Object> response;
    private Instant createdAt;

}
