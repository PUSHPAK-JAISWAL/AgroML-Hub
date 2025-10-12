package com.AgroHub_ML.backend.DTOS;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.RequiredArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@RequiredArgsConstructor
public class AuthResponse {

    private String token;
    private String tokenType = "Bearer";

}
