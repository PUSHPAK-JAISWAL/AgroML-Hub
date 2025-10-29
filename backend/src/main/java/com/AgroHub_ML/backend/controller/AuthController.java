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
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

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

    @GetMapping("/me")
    public ResponseEntity<?> me(@AuthenticationPrincipal UserDetails ud) {
        if(ud == null || ud.getUsername() == null) {
            return ResponseEntity.status(401).body(Map.of("error","Unauthorized"));
        }

        var userOpt = userService.findByEmail(ud.getUsername());
        if(userOpt.isEmpty()) {
            return ResponseEntity.status(401).body(Map.of("error","User not found"));
        }

        var u = userOpt.get();
        UserDto dto = new UserDto();
        dto.setId(u.getId());
        dto.setEmail(u.getEmail());
        dto.setRoles(u.getRoles());

        boolean isAdmin = u.getRoles() != null && u.getRoles().stream().anyMatch(r -> r.equalsIgnoreCase("ADMIN") || r.equalsIgnoreCase("ROLE_ADMIN"));

        return ResponseEntity.ok(Map.of(
            "user",dto,
                "isAdmin",isAdmin
        ));
    }

    @GetMapping("/is-admin")
    public ResponseEntity<?> isAdmin(@AuthenticationPrincipal UserDetails ud) {
        if(ud == null || ud.getUsername() == null) {
            return ResponseEntity.status(401).body(Map.of("isAdmin",false));
        }

        var userOpt = userService.findByEmail(ud.getUsername());
        if(userOpt.isEmpty()) {
            return ResponseEntity.status(401).body(Map.of("isAdmin",false));
        }

        var u = userOpt.get();
        boolean isAdmin = u.getRoles() != null && u.getRoles().stream().anyMatch(r-> r.equalsIgnoreCase("ADMIN") || r.equalsIgnoreCase("ROLE_ADMIN"));
        return ResponseEntity.ok(Map.of("isAdmin",isAdmin));
    }

}
