import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card } from "@/components/ui/card";
import { ArrowLeft } from "lucide-react";
import { toast } from "sonner";
import { api } from "@/lib/api";
import PredictionResults from "@/components/PredictionResults";

const PredictSeeds = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any>(null);
  
  const [formData, setFormData] = useState({
    Area: "",
    Perimeter: "",
    Compactness: "",
    length_of_kernel: "",
    width_of_kernel: "",
    asymetric_coef: "",
    length_of_kernel_groove: "",
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const sample = Object.fromEntries(
        Object.entries(formData).map(([key, value]) => [key, Number(value)])
      );

      const response = await api.predict({
        modelPath: "/seeds/predict",
        sample,
      });

      setResults(response);
      toast.success("Prediction completed!");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Prediction failed";
      toast.error(message);
      console.error("Prediction error:", error);
    } finally {
      setLoading(false);
    }
  };

  if (results) {
    return <PredictionResults results={results} onBack={() => setResults(null)} modelType="seeds" />;
  }

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

      <main className="container mx-auto px-4 py-8 max-w-3xl">
        <h1 className="text-3xl font-bold text-foreground mb-2">Wheat Seeds Classification</h1>
        <p className="text-muted-foreground mb-8">
          Enter geometric measurements of wheat kernels for variety classification
        </p>

        <Card className="p-6 shadow-medium">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <Label htmlFor="Area">Area</Label>
                <Input
                  id="Area"
                  type="number"
                  step="0.01"
                  value={formData.Area}
                  onChange={(e) => setFormData({ ...formData, Area: e.target.value })}
                  placeholder="15.26"
                  required
                />
              </div>

              <div>
                <Label htmlFor="Perimeter">Perimeter</Label>
                <Input
                  id="Perimeter"
                  type="number"
                  step="0.01"
                  value={formData.Perimeter}
                  onChange={(e) => setFormData({ ...formData, Perimeter: e.target.value })}
                  placeholder="14.84"
                  required
                />
              </div>

              <div>
                <Label htmlFor="Compactness">Compactness</Label>
                <Input
                  id="Compactness"
                  type="number"
                  step="0.001"
                  value={formData.Compactness}
                  onChange={(e) => setFormData({ ...formData, Compactness: e.target.value })}
                  placeholder="0.871"
                  required
                />
              </div>

              <div>
                <Label htmlFor="length_of_kernel">Kernel Length</Label>
                <Input
                  id="length_of_kernel"
                  type="number"
                  step="0.001"
                  value={formData.length_of_kernel}
                  onChange={(e) => setFormData({ ...formData, length_of_kernel: e.target.value })}
                  placeholder="5.763"
                  required
                />
              </div>

              <div>
                <Label htmlFor="width_of_kernel">Kernel Width</Label>
                <Input
                  id="width_of_kernel"
                  type="number"
                  step="0.001"
                  value={formData.width_of_kernel}
                  onChange={(e) => setFormData({ ...formData, width_of_kernel: e.target.value })}
                  placeholder="3.312"
                  required
                />
              </div>

              <div>
                <Label htmlFor="asymetric_coef">Asymmetric Coefficient</Label>
                <Input
                  id="asymetric_coef"
                  type="number"
                  step="0.001"
                  value={formData.asymetric_coef}
                  onChange={(e) => setFormData({ ...formData, asymetric_coef: e.target.value })}
                  placeholder="2.221"
                  required
                />
              </div>

              <div>
                <Label htmlFor="length_of_kernel_groove">Kernel Groove Length</Label>
                <Input
                  id="length_of_kernel_groove"
                  type="number"
                  step="0.01"
                  value={formData.length_of_kernel_groove}
                  onChange={(e) => setFormData({ ...formData, length_of_kernel_groove: e.target.value })}
                  placeholder="5.22"
                  required
                />
              </div>
            </div>

            <Button type="submit" className="w-full" disabled={loading}>
              {loading ? "Predicting..." : "Run Prediction"}
            </Button>
          </form>
        </Card>
      </main>
    </div>
  );
};

export default PredictSeeds;
