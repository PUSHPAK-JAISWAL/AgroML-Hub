package com.AgroHub_ML.backend.DTOS;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.RequiredArgsConstructor;

import java.util.Map;

@Data
public class PredictionResponse {
    private Map<String,Object> result;
}
