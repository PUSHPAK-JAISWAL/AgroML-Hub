import { Button } from "@/components/ui/button";
import { ArrowLeft } from "lucide-react";
import PredictionReport from "./PredictionReport";

interface PredictionResultsProps {
  results: any;
  onBack: () => void;
  modelType: "forest" | "seeds" | "mushrooms";
}

const PredictionResults = ({ results, onBack, modelType }: PredictionResultsProps) => {
  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card shadow-soft">
        <div className="container mx-auto px-4 py-4">
          <Button variant="ghost" onClick={onBack}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Form
          </Button>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-7xl">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground mb-2">Prediction Results</h1>
          <p className="text-muted-foreground">
            Multi-model ensemble prediction for {modelType} classification
          </p>
        </div>

        <PredictionReport results={results} modelType={modelType} />
      </main>
    </div>
  );
};

export default PredictionResults;
