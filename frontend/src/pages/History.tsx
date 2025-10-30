import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ArrowLeft } from "lucide-react";
import { api, PredictionHistory } from "@/lib/api";
import { toast } from "sonner";

const History = () => {
  const navigate = useNavigate();
  const [history, setHistory] = useState<PredictionHistory[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadHistory = async () => {
      try {
        const data = await api.getPredictionHistory();
        setHistory(data);
      } catch (error) {
        toast.error("Failed to load prediction history");
      } finally {
        setLoading(false);
      }
    };

    loadHistory();
  }, []);

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const getModelName = (endpoint: string) => {
    if (endpoint.includes("forest")) return "Forest Covertype";
    if (endpoint.includes("seeds")) return "Wheat Seeds";
    if (endpoint.includes("mushrooms")) return "Mushroom";
    return endpoint;
  };

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card shadow-soft">
        <div className="container mx-auto px-4 py-4">
          <Button variant="ghost" onClick={() => navigate("/dashboard")}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Dashboard
          </Button>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-6xl">
        <h1 className="text-3xl font-bold text-foreground mb-8">Prediction History</h1>

        {loading ? (
          <p className="text-muted-foreground">Loading history...</p>
        ) : history.length === 0 ? (
          <Card className="p-8 text-center">
            <p className="text-muted-foreground">No predictions yet. Start making predictions!</p>
            <Button className="mt-4" onClick={() => navigate("/dashboard")}>
              Go to Dashboard
            </Button>
          </Card>
        ) : (
          <div className="space-y-4">
            {history.map((item, index) => (
              <Card key={index} className="p-6 shadow-soft">
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <h3 className="text-lg font-semibold text-card-foreground">
                      {getModelName(item.endpoint)}
                    </h3>
                    <p className="text-sm text-muted-foreground">
                      {formatDate(item.createdAt)}
                    </p>
                  </div>
                </div>

                <div className="grid md:grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="font-medium text-card-foreground mb-2">Input Sample:</p>
                    <pre className="bg-muted p-3 rounded text-xs overflow-auto max-h-40">
                      {JSON.stringify(item.input.sample, null, 2)}
                    </pre>
                  </div>

                  <div>
                    <p className="font-medium text-card-foreground mb-2">Predictions:</p>
                    <div className="space-y-1">
                      {Object.entries(item.response).map(([model, data]: [string, any]) => (
                        <div key={model} className="bg-muted p-2 rounded text-xs">
                          <span className="font-medium">{model}:</span>{" "}
                          {data.label || `Class ${data.prediction}`}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        )}
      </main>
    </div>
  );
};

export default History;
