import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { CheckCircle2, AlertCircle } from "lucide-react";

interface ModelPrediction {
  prediction: number;
  label?: string;
  probabilities?: number[];
}

interface PredictionReportProps {
  results: Record<string, ModelPrediction>;
  modelType: "forest" | "seeds" | "mushrooms";
}

const PredictionReport = ({ results, modelType }: PredictionReportProps) => {
  const models = Object.keys(results);
  
  const getModelName = (key: string) => {
    const names: Record<string, string> = {
      randomforest: "Random Forest",
      xgboost: "XGBoost",
      tensorflow: "TensorFlow",
      quantized: "Quantized Model",
    };
    return names[key] || key;
  };


  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return "text-emerald-600";
    if (confidence >= 60) return "text-amber-600";
    return "text-rose-600";
  };

  const getTopProbabilities = (probs: number[]) => {
    return probs
      .map((p, i) => ({ class: i + 1, prob: p }))
      .sort((a, b) => b.prob - a.prob)
      .slice(0, 3);
  };

  return (
    <div className="space-y-8">
      {/* Individual Model Results */}
      <div>
        <h3 className="text-xl font-semibold text-foreground mb-4">Individual Model Results</h3>
        <div className="grid md:grid-cols-2 gap-6">
          {models.map((modelKey) => {
            const modelData = results[modelKey];
            const probabilities = modelData.probabilities || [];
            const maxProb = Math.max(...probabilities) * 100;

            return (
              <Card key={modelKey} className="p-6 shadow-medium hover:shadow-strong transition-shadow">
                <div className="flex items-center justify-between mb-4">
                  <h4 className="text-lg font-semibold text-card-foreground">
                    {getModelName(modelKey)}
                  </h4>
                </div>

                <div className="space-y-4">
                  <div className="p-4 rounded-lg bg-muted/50">
                    <p className="text-sm text-muted-foreground mb-1">Prediction</p>
                    <p className="text-2xl font-bold text-foreground">
                      {modelData.label || `Class ${modelData.prediction}`}
                    </p>
                  </div>

                  {probabilities.length > 0 && (
                    <div>
                      <div className="flex items-center justify-between mb-3">
                        <p className="text-sm font-medium text-muted-foreground">
                          Confidence
                        </p>
                        <p className={`text-sm font-bold ${getConfidenceColor(maxProb)}`}>
                          {maxProb.toFixed(1)}%
                        </p>
                      </div>
                      <Progress value={maxProb} className="h-2 mb-4" />
                      
                      <div className="space-y-2">
                        <p className="text-xs text-muted-foreground mb-2">Top Probabilities:</p>
                        {getTopProbabilities(probabilities).map(({ class: cls, prob }) => (
                          <div key={cls} className="flex items-center gap-3">
                            <span className="text-xs text-muted-foreground w-20">
                              Class {cls}
                            </span>
                            <div className="flex-1 bg-muted rounded-full h-1.5 overflow-hidden">
                              <div
                                className="bg-accent h-full rounded-full transition-all"
                                style={{ width: `${prob * 100}%` }}
                              />
                            </div>
                            <span className="text-xs font-medium text-foreground w-12 text-right">
                              {(prob * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </Card>
            );
          })}
        </div>
      </div>

      {/* Summary Statistics */}
      <Card className="p-6 shadow-medium">
        <h3 className="text-lg font-semibold text-foreground mb-4">Prediction Summary</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-4 rounded-lg bg-muted/50">
            <p className="text-2xl font-bold text-primary">{models.length}</p>
            <p className="text-xs text-muted-foreground mt-1">Models Used</p>
          </div>
          <div className="text-center p-4 rounded-lg bg-muted/50">
            <p className="text-2xl font-bold text-primary">
              {Math.max(...models.map(m => Math.max(...(results[m].probabilities || [0])) * 100)).toFixed(0)}%
            </p>
            <p className="text-xs text-muted-foreground mt-1">Max Confidence</p>
          </div>
          <div className="text-center p-4 rounded-lg bg-muted/50">
            <p className="text-2xl font-bold text-primary capitalize">{modelType}</p>
            <p className="text-xs text-muted-foreground mt-1">Model Type</p>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default PredictionReport;
