package com.AgroHub_ML.backend.Entity;


import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.RequiredArgsConstructor;
import org.bson.types.ObjectId;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

import java.util.List;
import java.util.Set;

@Document(collection = "users")
@AllArgsConstructor
@RequiredArgsConstructor
@Data
public class User {

    @Id
    private ObjectId id;
    private String email;
    private String password;
    private Set<String> roles;
}
