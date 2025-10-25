package com.AgroHub_ML.backend.DTOS;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.RequiredArgsConstructor;

@Data
public class RegisterRequest {

    @Email @NotBlank
    private String email;

    @NotBlank
    private String password;
}
