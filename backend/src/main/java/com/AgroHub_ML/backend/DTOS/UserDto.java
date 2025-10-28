package com.AgroHub_ML.backend.DTOS;


import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.RequiredArgsConstructor;
import org.bson.types.ObjectId;

import java.util.List;
import java.util.Set;

@Data
public class UserDto {
    private ObjectId id;
    private String email;
    private Set<String> roles;
}
