package com.AgroHub_ML.backend.Entity;


import lombok.*;
import org.bson.types.ObjectId;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

import java.time.Instant;
import java.util.List;
import java.util.Set;

@Document(collection = "users")
@AllArgsConstructor
@RequiredArgsConstructor
@Data
@Builder
public class User {

    @Id
    private ObjectId id;
    private String email;
    private String password;
    private Set<String> roles;
    private Instant createdAt;
}
