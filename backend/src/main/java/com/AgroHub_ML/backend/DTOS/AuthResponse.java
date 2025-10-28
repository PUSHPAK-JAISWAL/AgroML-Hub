package com.AgroHub_ML.backend.DTOS;

import lombok.*;

@Data
@AllArgsConstructor
@RequiredArgsConstructor
public class AuthResponse {

    @NonNull
    private String token;

    private String tokenType = "Bearer";

}
