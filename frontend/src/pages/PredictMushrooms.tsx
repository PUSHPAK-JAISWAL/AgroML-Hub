import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Card } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ArrowLeft } from "lucide-react";
import { toast } from "sonner";
import { api } from "@/lib/api";
import PredictionResults from "@/components/PredictionResults";

const PredictMushrooms = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any>(null);
  
  const [formData, setFormData] = useState({
    "cap-shape": "",
    "cap-surface": "",
    "cap-color": "",
    "bruises": "",
    "odor": "",
    "gill-attachment": "",
    "gill-spacing": "",
    "gill-size": "",
    "gill-color": "",
    "stalk-shape": "",
    "stalk-root": "",
    "stalk-surface-above-ring": "",
    "stalk-surface-below-ring": "",
    "stalk-color-above-ring": "",
    "stalk-color-below-ring": "",
    "veil-type": "",
    "veil-color": "",
    "ring-number": "",
    "ring-type": "",
    "spore-print-color": "",
    "population": "",
    "habitat": "",
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await api.predict({
        modelPath: "/mushrooms/predict",
        sample: formData,
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
    return <PredictionResults results={results} onBack={() => setResults(null)} modelType="mushrooms" />;
  }

  const updateField = (field: string, value: string) => {
    setFormData({ ...formData, [field]: value });
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

      <main className="container mx-auto px-4 py-8 max-w-5xl">
        <h1 className="text-3xl font-bold text-foreground mb-2">Mushroom Classification</h1>
        <p className="text-muted-foreground mb-8">
          Enter physical characteristics to classify mushroom edibility
        </p>

        <Card className="p-6 shadow-medium">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid md:grid-cols-3 gap-4">
              <div>
                <Label>Cap Shape</Label>
                <Select value={formData["cap-shape"]} onValueChange={(v) => updateField("cap-shape", v)}>
                  <SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="bell">Bell</SelectItem>
                    <SelectItem value="conical">Conical</SelectItem>
                    <SelectItem value="convex">Convex</SelectItem>
                    <SelectItem value="flat">Flat</SelectItem>
                    <SelectItem value="knobbed">Knobbed</SelectItem>
                    <SelectItem value="sunken">Sunken</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label>Cap Surface</Label>
                <Select value={formData["cap-surface"]} onValueChange={(v) => updateField("cap-surface", v)}>
                  <SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="fibrous">Fibrous</SelectItem>
                    <SelectItem value="grooves">Grooves</SelectItem>
                    <SelectItem value="scaly">Scaly</SelectItem>
                    <SelectItem value="smooth">Smooth</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label>Cap Color</Label>
                <Select value={formData["cap-color"]} onValueChange={(v) => updateField("cap-color", v)}>
                  <SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="brown">Brown</SelectItem>
                    <SelectItem value="buff">Buff</SelectItem>
                    <SelectItem value="cinnamon">Cinnamon</SelectItem>
                    <SelectItem value="gray">Gray</SelectItem>
                    <SelectItem value="green">Green</SelectItem>
                    <SelectItem value="pink">Pink</SelectItem>
                    <SelectItem value="purple">Purple</SelectItem>
                    <SelectItem value="red">Red</SelectItem>
                    <SelectItem value="white">White</SelectItem>
                    <SelectItem value="yellow">Yellow</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label>Bruises</Label>
                <Select value={formData["bruises"]} onValueChange={(v) => updateField("bruises", v)}>
                  <SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="bruises">Bruises</SelectItem>
                    <SelectItem value="no">No</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label>Odor</Label>
                <Select value={formData["odor"]} onValueChange={(v) => updateField("odor", v)}>
                  <SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="almond">Almond</SelectItem>
                    <SelectItem value="anise">Anise</SelectItem>
                    <SelectItem value="creosote">Creosote</SelectItem>
                    <SelectItem value="fishy">Fishy</SelectItem>
                    <SelectItem value="foul">Foul</SelectItem>
                    <SelectItem value="musty">Musty</SelectItem>
                    <SelectItem value="none">None</SelectItem>
                    <SelectItem value="pungent">Pungent</SelectItem>
                    <SelectItem value="spicy">Spicy</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label>Gill Attachment</Label>
                <Select value={formData["gill-attachment"]} onValueChange={(v) => updateField("gill-attachment", v)}>
                  <SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="attached">Attached</SelectItem>
                    <SelectItem value="descending">Descending</SelectItem>
                    <SelectItem value="free">Free</SelectItem>
                    <SelectItem value="notched">Notched</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label>Gill Spacing</Label>
                <Select value={formData["gill-spacing"]} onValueChange={(v) => updateField("gill-spacing", v)}>
                  <SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="close">Close</SelectItem>
                    <SelectItem value="crowded">Crowded</SelectItem>
                    <SelectItem value="distant">Distant</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label>Gill Size</Label>
                <Select value={formData["gill-size"]} onValueChange={(v) => updateField("gill-size", v)}>
                  <SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="broad">Broad</SelectItem>
                    <SelectItem value="narrow">Narrow</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label>Gill Color</Label>
                <Select value={formData["gill-color"]} onValueChange={(v) => updateField("gill-color", v)}>
                  <SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="black">Black</SelectItem>
                    <SelectItem value="brown">Brown</SelectItem>
                    <SelectItem value="buff">Buff</SelectItem>
                    <SelectItem value="chocolate">Chocolate</SelectItem>
                    <SelectItem value="gray">Gray</SelectItem>
                    <SelectItem value="green">Green</SelectItem>
                    <SelectItem value="orange">Orange</SelectItem>
                    <SelectItem value="pink">Pink</SelectItem>
                    <SelectItem value="purple">Purple</SelectItem>
                    <SelectItem value="red">Red</SelectItem>
                    <SelectItem value="white">White</SelectItem>
                    <SelectItem value="yellow">Yellow</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label>Stalk Shape</Label>
                <Select value={formData["stalk-shape"]} onValueChange={(v) => updateField("stalk-shape", v)}>
                  <SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="enlarging">Enlarging</SelectItem>
                    <SelectItem value="tapering">Tapering</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label>Stalk Root</Label>
                <Select value={formData["stalk-root"]} onValueChange={(v) => updateField("stalk-root", v)}>
                  <SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="bulbous">Bulbous</SelectItem>
                    <SelectItem value="club">Club</SelectItem>
                    <SelectItem value="cup">Cup</SelectItem>
                    <SelectItem value="equal">Equal</SelectItem>
                    <SelectItem value="rhizomorphs">Rhizomorphs</SelectItem>
                    <SelectItem value="rooted">Rooted</SelectItem>
                    <SelectItem value="missing">Missing</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label>Stalk Surface Above Ring</Label>
                <Select value={formData["stalk-surface-above-ring"]} onValueChange={(v) => updateField("stalk-surface-above-ring", v)}>
                  <SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="fibrous">Fibrous</SelectItem>
                    <SelectItem value="scaly">Scaly</SelectItem>
                    <SelectItem value="silky">Silky</SelectItem>
                    <SelectItem value="smooth">Smooth</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label>Stalk Surface Below Ring</Label>
                <Select value={formData["stalk-surface-below-ring"]} onValueChange={(v) => updateField("stalk-surface-below-ring", v)}>
                  <SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="fibrous">Fibrous</SelectItem>
                    <SelectItem value="scaly">Scaly</SelectItem>
                    <SelectItem value="silky">Silky</SelectItem>
                    <SelectItem value="smooth">Smooth</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label>Stalk Color Above Ring</Label>
                <Select value={formData["stalk-color-above-ring"]} onValueChange={(v) => updateField("stalk-color-above-ring", v)}>
                  <SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="brown">Brown</SelectItem>
                    <SelectItem value="buff">Buff</SelectItem>
                    <SelectItem value="cinnamon">Cinnamon</SelectItem>
                    <SelectItem value="gray">Gray</SelectItem>
                    <SelectItem value="orange">Orange</SelectItem>
                    <SelectItem value="pink">Pink</SelectItem>
                    <SelectItem value="red">Red</SelectItem>
                    <SelectItem value="white">White</SelectItem>
                    <SelectItem value="yellow">Yellow</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label>Stalk Color Below Ring</Label>
                <Select value={formData["stalk-color-below-ring"]} onValueChange={(v) => updateField("stalk-color-below-ring", v)}>
                  <SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="brown">Brown</SelectItem>
                    <SelectItem value="buff">Buff</SelectItem>
                    <SelectItem value="cinnamon">Cinnamon</SelectItem>
                    <SelectItem value="gray">Gray</SelectItem>
                    <SelectItem value="orange">Orange</SelectItem>
                    <SelectItem value="pink">Pink</SelectItem>
                    <SelectItem value="red">Red</SelectItem>
                    <SelectItem value="white">White</SelectItem>
                    <SelectItem value="yellow">Yellow</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label>Veil Type</Label>
                <Select value={formData["veil-type"]} onValueChange={(v) => updateField("veil-type", v)}>
                  <SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="partial">Partial</SelectItem>
                    <SelectItem value="universal">Universal</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label>Veil Color</Label>
                <Select value={formData["veil-color"]} onValueChange={(v) => updateField("veil-color", v)}>
                  <SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="brown">Brown</SelectItem>
                    <SelectItem value="orange">Orange</SelectItem>
                    <SelectItem value="white">White</SelectItem>
                    <SelectItem value="yellow">Yellow</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label>Ring Number</Label>
                <Select value={formData["ring-number"]} onValueChange={(v) => updateField("ring-number", v)}>
                  <SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="none">None</SelectItem>
                    <SelectItem value="one">One</SelectItem>
                    <SelectItem value="two">Two</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label>Ring Type</Label>
                <Select value={formData["ring-type"]} onValueChange={(v) => updateField("ring-type", v)}>
                  <SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="cobwebby">Cobwebby</SelectItem>
                    <SelectItem value="evanescent">Evanescent</SelectItem>
                    <SelectItem value="flaring">Flaring</SelectItem>
                    <SelectItem value="large">Large</SelectItem>
                    <SelectItem value="none">None</SelectItem>
                    <SelectItem value="pendant">Pendant</SelectItem>
                    <SelectItem value="sheathing">Sheathing</SelectItem>
                    <SelectItem value="zone">Zone</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label>Spore Print Color</Label>
                <Select value={formData["spore-print-color"]} onValueChange={(v) => updateField("spore-print-color", v)}>
                  <SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="black">Black</SelectItem>
                    <SelectItem value="brown">Brown</SelectItem>
                    <SelectItem value="buff">Buff</SelectItem>
                    <SelectItem value="chocolate">Chocolate</SelectItem>
                    <SelectItem value="green">Green</SelectItem>
                    <SelectItem value="orange">Orange</SelectItem>
                    <SelectItem value="purple">Purple</SelectItem>
                    <SelectItem value="white">White</SelectItem>
                    <SelectItem value="yellow">Yellow</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label>Population</Label>
                <Select value={formData["population"]} onValueChange={(v) => updateField("population", v)}>
                  <SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="abundant">Abundant</SelectItem>
                    <SelectItem value="clustered">Clustered</SelectItem>
                    <SelectItem value="numerous">Numerous</SelectItem>
                    <SelectItem value="scattered">Scattered</SelectItem>
                    <SelectItem value="several">Several</SelectItem>
                    <SelectItem value="solitary">Solitary</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label>Habitat</Label>
                <Select value={formData["habitat"]} onValueChange={(v) => updateField("habitat", v)}>
                  <SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="grasses">Grasses</SelectItem>
                    <SelectItem value="leaves">Leaves</SelectItem>
                    <SelectItem value="meadows">Meadows</SelectItem>
                    <SelectItem value="paths">Paths</SelectItem>
                    <SelectItem value="urban">Urban</SelectItem>
                    <SelectItem value="waste">Waste</SelectItem>
                    <SelectItem value="woods">Woods</SelectItem>
                  </SelectContent>
                </Select>
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

export default PredictMushrooms;
