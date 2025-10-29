package com.AgroHub_ML.backend.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;
import org.springframework.web.filter.CorsFilter;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

import java.util.List;

@Configuration
public class CorsConfig {

    @Bean
    public CorsFilter corsFilter() {
        CorsConfiguration config = new CorsConfiguration();

        // Development: allow any origin pattern. For production, replace with explicit origins.
        config.setAllowedOriginPatterns(List.of("*"));

        // Allowed HTTP methods
        config.setAllowedMethods(List.of("GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"));

        // Allow any header from client
        config.setAllowedHeaders(List.of("*"));

        // Allow credentials (cookies / Authorization header) — keep if clients send Authorization header
        config.setAllowCredentials(true);

        // Expose headers to the browser (if you set custom headers)
        config.setExposedHeaders(List.of("Authorization", "Content-Type", "Location"));

        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        // Apply to all paths
        source.registerCorsConfiguration("/**", config);

        return new CorsFilter(source);
    }

    // Optional: keep MVC-level CORS mapping for controllers (consistent behavior)
    @Bean
    public WebMvcConfigurer webMvcConfigurer() {
        return new WebMvcConfigurer() {
            // nothing required here because CorsFilter handles it,
            // but leaving this method allows future fine tuning.
        };
    }
}
