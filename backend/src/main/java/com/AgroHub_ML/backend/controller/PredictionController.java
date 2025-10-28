package com.AgroHub_ML.backend.controller;

import com.AgroHub_ML.backend.DTOS.PredictionRequest;
import com.AgroHub_ML.backend.Entity.User;
import com.AgroHub_ML.backend.Repositories.UserRepository;
import com.AgroHub_ML.backend.service.PredictionService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/predictions")
@RequiredArgsConstructor
public class PredictionController {

    @Autowired
    private PredictionService predictionService;

    @Autowired
    private UserRepository userRepository;

    @PostMapping
    public ResponseEntity<?> predict(@AuthenticationPrincipal UserDetails ud,
                                     @Valid @RequestBody PredictionRequest req) {
        var userOpt = userRepository.findByEmail(ud.getUsername());
        if(userOpt.isEmpty()) return ResponseEntity.status(401).body(Map.of("error","User not found"));
        User user = userOpt.get();

        Map<String,Object> forwarded = Map.of("sample",req.getSample());
        Map<String,Object> result = predictionService.predictAndSave(user.getId(), req.getModelPath(), forwarded,user.getEmail());
        return ResponseEntity.ok(result);
    }

    @GetMapping("/history")
    public ResponseEntity<?> history(@AuthenticationPrincipal UserDetails ud) {
        var userOpt = userRepository.findByEmail(ud.getUsername());
        if(userOpt.isEmpty()) return ResponseEntity.status(401).body(Map.of("error","User not found"));
        var history = predictionService.getHistoryForUser(userOpt.get().getId());
        return ResponseEntity.ok(history);
    }

}
