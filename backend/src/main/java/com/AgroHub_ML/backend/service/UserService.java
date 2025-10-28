package com.AgroHub_ML.backend.service;

import com.AgroHub_ML.backend.DTOS.RegisterRequest;
import com.AgroHub_ML.backend.Entity.User;
import com.AgroHub_ML.backend.Repositories.UserRepository;
import lombok.RequiredArgsConstructor;
import org.bson.types.ObjectId;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Optional;
import java.util.Set;

@Service
@RequiredArgsConstructor
public class UserService {

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private PasswordEncoder passwordEncoder;

    public User registerUser(RegisterRequest req) {
        if(userRepository.existsByEmail(req.getEmail())) {
            throw new IllegalArgumentException("Email Already Used.");
        }

        User u = User.builder()
                .email(req.getEmail())
                .password(passwordEncoder.encode(req.getPassword()))
                .roles(Set.of("ROLE_USER"))
                .createdAt(Instant.now())
                .build();

        return userRepository.save(u);
    }

    public List<User> findAll() {
        return userRepository.findAll();
    }

    public Optional<User> findById(ObjectId id) {
        return userRepository.findById(id);
    }

    public Optional<User> findByEmail(String email) {
        return userRepository.findByEmail(email);
    }

    public User createUser(User user) {
        if(userRepository.existsByEmail(user.getEmail())) throw new IllegalArgumentException("Email exists");
        user.setPassword(passwordEncoder.encode(user.getPassword()));
        user.setCreatedAt(Instant.now());
        return userRepository.save(user);
    }

    public User updateUser(ObjectId id, User update) {
        var u = userRepository.findById(id).orElseThrow(()-> new NoSuchElementException("User not found"));
        if(update.getPassword() != null && !update.getPassword().isBlank()) {
            u.setPassword(passwordEncoder.encode(update.getPassword()));
        }

        if(update.getRoles() != null) u.setRoles(update.getRoles());
        return userRepository.save(u);
    }

    public void deleteUser(ObjectId id) {
        userRepository.deleteById(id);
    }
}
