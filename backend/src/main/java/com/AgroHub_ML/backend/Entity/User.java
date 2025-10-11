package com.AgroHub_ML.backend.Entity;


import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.RequiredArgsConstructor;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

import java.util.List;

@Document(collection = "users")
@AllArgsConstructor
@RequiredArgsConstructor
@NoArgsConstructor
@Data
public class User {

    @Id
    private String id;
    private String email;
    private String password;
    private List<String> roles;
}
