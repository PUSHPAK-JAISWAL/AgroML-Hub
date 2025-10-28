package com.AgroHub_ML.backend.service;

import com.AgroHub_ML.backend.DTOS.AuthRequest;
import com.AgroHub_ML.backend.DTOS.AuthResponse;
import com.AgroHub_ML.backend.utils.JwtUtil;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class AuthService {

    @Autowired
    private AuthenticationManager authenticationManager;

    @Autowired
    private JwtUtil jwtUtil;

    @Autowired
    private UserDetailsService userDetailsService;

    public AuthResponse login(AuthRequest req) {
        var authToken = new UsernamePasswordAuthenticationToken(req.getEmail(),req.getPassword());
        authenticationManager.authenticate(authToken);
        UserDetails ud = userDetailsService.loadUserByUsername(req.getEmail());
        String token = jwtUtil.generateToken(ud.getUsername());
        return new AuthResponse(token);
    }

}
