package com.AgroHub_ML.backend.service;
//this is a unnessesary comment i am adding to showcase owner ship of this project.
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.mail.MessagingException;
import jakarta.mail.internet.MimeMessage;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.mail.javamail.MimeMessageHelper;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.*;

@Service
@RequiredArgsConstructor
@Slf4j
public class EmailService {

    private final JavaMailSender mailSender;
    private final ObjectMapper objectMapper = new ObjectMapper();

    /**
     * Backward-compatible simple plain-text sender (keeps your existing code working).
     */
    public void sendPredictionEmail(String to, String subject, String body) {
        try {
            MimeMessage msg = mailSender.createMimeMessage();
            MimeMessageHelper helper = new MimeMessageHelper(msg, false, "UTF-8");
            helper.setTo(to);
            helper.setSubject(subject);
            helper.setText(body, false);
            mailSender.send(msg);
        } catch (Exception ex) {
            log.error("Failed to send plain text email to {}: {}", to, ex.getMessage(), ex);
        }
    }

    /**
     * Send a nicely formatted HTML report for a prediction.
     *
     * @param to       recipient email
     * @param subject  email subject
     * @param input    the input payload forwarded to the model server (e.g. {"sample": {...}})
     * @param results  model results (expected shape: { "randomforest": {...}, "xgboost": {...}, ... })
     */
    @Async
    public void sendPredictionReport(String to,
                                     String subject,
                                     Map<String, Object> input,
                                     Map<String, Object> results) {
        try {
            String html = buildHtmlReport(input, results);
            MimeMessage msg = mailSender.createMimeMessage();
            MimeMessageHelper helper = new MimeMessageHelper(msg, true, "UTF-8"); // multipart = true
            helper.setTo(to);
            helper.setSubject(subject);
            helper.setText(html, true); // true = isHtml
            mailSender.send(msg);
            log.info("Sent prediction report to {}", to);
        } catch (MessagingException | RuntimeException ex) {
            log.error("Failed to send prediction report to {}: {}", to, ex.getMessage(), ex);
        }
    }

    // --------------------------
    // Helper: create HTML content
    // --------------------------
    private String buildHtmlReport(Map<String, Object> input, Map<String, Object> results) {
        String title = "Prediction Report";
        String timestamp = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")
                .withZone(ZoneId.systemDefault())
                .format(Instant.now());

        StringBuilder sb = new StringBuilder();
        sb.append("<!doctype html><html><head><meta charset='utf-8'/>");
        sb.append("<meta name='viewport' content='width=device-width, initial-scale=1'/>");
        sb.append("<style>");
        // Basic, email-friendly inline styles:
        sb.append("body{font-family:Arial,Helvetica,sans-serif;color:#222;margin:0;padding:0;background:#f6f7fb}");
        sb.append(".container{max-width:700px;margin:24px auto;background:white;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,0.08);overflow:hidden}");
        sb.append(".header{background:#2b6cb0;color:white;padding:18px 24px}");
        sb.append(".header h1{margin:0;font-size:20px}");
        sb.append(".meta{padding:12px 24px;color:#555;font-size:13px;border-bottom:1px solid #eee}");
        sb.append(".section{padding:16px 24px;border-bottom:1px solid #f1f3f6}");
        sb.append(".section h2{margin:0 0 10px 0;font-size:16px;color:#333}");
        sb.append("table{width:100%;border-collapse:collapse;font-size:13px}");
        sb.append("th,td{padding:8px;text-align:left;border-bottom:1px solid #eee}");
        sb.append(".badge{display:inline-block;padding:4px 8px;border-radius:12px;background:#edf2ff;color:#2b6cb0;font-weight:600;font-size:12px}");
        sb.append(".model-card{border:1px solid #e6eefc;border-radius:6px;padding:12px;margin-bottom:12px;background:#fbfdff}");
        sb.append(".small{font-size:12px;color:#666}");
        sb.append(".footer{padding:12px 24px;color:#888;font-size:12px}");
        sb.append("</style></head><body>");
        sb.append("<div class='container'>");
        sb.append("<div class='header'><h1>").append(escapeHtml(title)).append("</h1></div>");
        sb.append("<div class='meta'><span class='badge'>AgroML Hub</span> &nbsp; Report generated: ").append(escapeHtml(timestamp)).append("</div>");

        // Input section
        sb.append("<div class='section'>");
        sb.append("<h2>Input (submitted)</h2>");
        sb.append(renderInputTable(input));
        sb.append("</div>");

        // Results section: iterate models
        sb.append("<div class='section'>");
        sb.append("<h2>Model results</h2>");
        if (results == null || results.isEmpty()) {
            sb.append("<div class='small'>No results available.</div>");
        } else {
            // prefer stable order: randomforest, xgboost, tensorflow, quantized
            List<String> prefer = Arrays.asList("randomforest", "xgboost", "tensorflow", "quantized");
            Set<String> keys = new LinkedHashSet<>(prefer);
            keys.addAll(results.keySet()); // keep other keys afterwards
            for (String k : keys) {
                if (!results.containsKey(k)) continue;
                Object modelObj = results.get(k);
                sb.append(renderModelCard(k, modelObj));
            }
        }
        sb.append("</div>");

        // Footer
        sb.append("<div class='footer'>");
        sb.append("This is an automated message from <strong>AgroML Hub</strong>. ");
        sb.append("If you did not request this, please contact support. ");
        sb.append("</div>");

        sb.append("</div>"); // container
        sb.append("</body></html>");
        return sb.toString();
    }

    // Render the input map as a key/value table
    private String renderInputTable(Map<String, Object> input) {
        if (input == null || input.isEmpty()) return "<div class='small'>No input provided</div>";
        // Some payloads use {"sample": {..}} — flatten if present
        Object sample = input.getOrDefault("sample", input);
        Map<String, Object> map;
        if (sample instanceof Map) {
            //noinspection unchecked
            map = (Map<String, Object>) sample;
        } else {
            // fallback: try to convert input to JSON string
            return "<pre class='small'>" + escapeHtml(toJsonSafe(input)) + "</pre>";
        }

        StringBuilder sb = new StringBuilder();
        sb.append("<table><thead><tr><th>Feature</th><th>Value</th></tr></thead><tbody>");
        map.forEach((k, v) -> {
            sb.append("<tr><td>").append(escapeHtml(String.valueOf(k))).append("</td>");
            sb.append("<td>").append(escapeHtml(String.valueOf(v))).append("</td></tr>");
        });
        sb.append("</tbody></table>");
        return sb.toString();
    }

    // Render a model result card (prediction + probabilities)
    private String renderModelCard(String modelKey, Object modelObj) {
        String displayName = prettifyModelName(modelKey);
        StringBuilder sb = new StringBuilder();
        sb.append("<div class='model-card'>");
        sb.append("<strong>").append(escapeHtml(displayName)).append("</strong>");

        if (modelObj instanceof Map) {
            //noinspection unchecked
            Map<String, Object> m = (Map<String, Object>) modelObj;
            Object pred = m.get("prediction");
            Object label = m.get("label");
            Object probs = m.get("probabilities");

            sb.append("<div class='small' style='margin-top:8px;'>");
            sb.append("Prediction: <strong>").append(escapeHtml(String.valueOf(pred))).append("</strong>");
            if (label != null) sb.append(" — <em>").append(escapeHtml(String.valueOf(label))).append("</em>");
            sb.append("</div>");

            // Probabilities rendering
            if (probs instanceof Collection) {
                Collection<?> probList = (Collection<?>) probs;
                sb.append("<div style='margin-top:8px;'><table><thead><tr><th>#</th><th>Probability</th></tr></thead><tbody>");
                int idx = 1;
                for (Object p : probList) {
                    sb.append("<tr><td>").append(idx++).append("</td>");
                    sb.append("<td>").append(escapeHtml(String.format(Locale.US, "%.4f", toDoubleSafe(p)))).append("</td></tr>");
                }
                sb.append("</tbody></table></div>");
            } else if (probs != null) {
                // fallback: just pretty-print
                sb.append("<pre class='small'>").append(escapeHtml(toJsonSafe(probs))).append("</pre>");
            }
        } else {
            // Not a map — pretty print whole value
            sb.append("<div class='small' style='margin-top:8px;'><pre>").append(escapeHtml(toJsonSafe(modelObj))).append("</pre></div>");
        }

        sb.append("</div>"); // model-card
        return sb.toString();
    }

    // Small helpers
    private String prettifyModelName(String key) {
        return switch (key.toLowerCase()) {
            case "randomforest" -> "Random Forest";
            case "xgboost" -> "XGBoost";
            case "tensorflow" -> "TensorFlow (Keras)";
            case "quantized" -> "TensorFlow (TFLite - quantized)";
            default -> key;
        };
    }

    private double toDoubleSafe(Object o) {
        if (o == null) return 0.0;
        if (o instanceof Number) return ((Number) o).doubleValue();
        try {
            return Double.parseDouble(String.valueOf(o));
        } catch (Exception ex) {
            return 0.0;
        }
    }

    private String toJsonSafe(Object o) {
        try {
            return objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(o);
        } catch (JsonProcessingException e) {
            return String.valueOf(o);
        }
    }

    /**
     * Very small HTML escaper for safety in emails (keeps it simple).
     */
    private String escapeHtml(String s) {
        if (s == null) return "";
        return s.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\"", "&quot;")
                .replace("'", "&#39;");
    }
}
