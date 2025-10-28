package com.AgroHub_ML.backend.Repositories;

import com.AgroHub_ML.backend.Entity.PredictionHistory;
import org.bson.types.ObjectId;
import org.springframework.data.mongodb.repository.MongoRepository;

import java.util.List;

public interface PredictionRepository extends MongoRepository<PredictionHistory, ObjectId> {
    List<PredictionHistory> findByUserIdOrderByCreatedAtDesc(ObjectId userId);
}
