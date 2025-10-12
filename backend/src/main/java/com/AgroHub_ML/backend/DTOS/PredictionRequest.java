package com.AgroHub_ML.backend.DTOS;

import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.RequiredArgsConstructor;

import java.util.Map;

@Data
@AllArgsConstructor
@NoArgsConstructor
@RequiredArgsConstructor
public class PredictionRequest {
    @NotNull
    private String modelPath;

    @NotNull
    private Map<String,Object> sample;
}
