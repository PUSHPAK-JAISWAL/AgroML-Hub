package com.AgroHub_ML.backend.DTOS;


import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.RequiredArgsConstructor;
import org.bson.types.ObjectId;

import java.util.List;

@Data
public class UserDto {
    private ObjectId id;
    private String email;
    private List<String> roles;
}
