package com.AgroHub_ML.backend.controller;

import com.AgroHub_ML.backend.DTOS.AuthRequest;
import com.AgroHub_ML.backend.DTOS.AuthResponse;
import com.AgroHub_ML.backend.DTOS.RegisterRequest;
import com.AgroHub_ML.backend.DTOS.UserDto;
import com.AgroHub_ML.backend.Entity.User;
import com.AgroHub_ML.backend.service.AuthService;
import com.AgroHub_ML.backend.service.UserService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/auth")
@RequiredArgsConstructor
public class AuthController {

    @Autowired
    private AuthService authService;

    @Autowired
    private UserService userService;

    @PostMapping("/register")
    public ResponseEntity<?> register(@Valid @RequestBody RegisterRequest req){
        User u = userService.registerUser(req);
        UserDto dto = new UserDto();
        dto.setId(u.getId());
        dto.setEmail(u.getEmail());
        dto.setRoles(u.getRoles());
        return ResponseEntity.ok(dto);
    }

    @PostMapping("/login")
    public ResponseEntity<AuthResponse> login(@Valid @RequestBody AuthRequest req) {
        AuthResponse res = authService.login(req);
        return ResponseEntity.ok(res);
    }

}
