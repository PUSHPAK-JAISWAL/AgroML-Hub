import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ArrowLeft } from "lucide-react";
import { toast } from "sonner";
import { api } from "@/lib/api";
import PredictionResults from "@/components/PredictionResults";

const PredictForest = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any>(null);
  
  const [formData, setFormData] = useState({
    Elevation: "",
    Aspect: "",
    Slope: "",
    Horizontal_Distance_To_Hydrology: "",
    Vertical_Distance_To_Hydrology: "",
    Horizontal_Distance_To_Roadways: "",
    Hillshade_9am: "",
    Hillshade_Noon: "",
    Hillshade_3pm: "",
    Horizontal_Distance_To_Fire_Points: "",
    Wilderness_Area: "",
    Soil_Type: "",
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const sample = {
        ...Object.fromEntries(
          Object.entries(formData).map(([key, value]) => [
            key,
            key === "Wilderness_Area" || key === "Soil_Type" ? value : Number(value)
          ])
        )
      };

      const response = await api.predict({
        modelPath: "/forest/predict",
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
    return <PredictionResults results={results} onBack={() => setResults(null)} modelType="forest" />;
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

      <main className="container mx-auto px-4 py-8 max-w-4xl">
        <h1 className="text-3xl font-bold text-foreground mb-2">Forest Covertype Prediction</h1>
        <p className="text-muted-foreground mb-8">
          Enter cartographic variables to predict the forest cover type
        </p>

        <Card className="p-6 shadow-medium">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <Label htmlFor="Elevation">Elevation (meters)</Label>
                <Input
                  id="Elevation"
                  type="number"
                  value={formData.Elevation}
                  onChange={(e) => setFormData({ ...formData, Elevation: e.target.value })}
                  placeholder="2596"
                  required
                />
              </div>

              <div>
                <Label htmlFor="Aspect">Aspect (degrees azimuth)</Label>
                <Input
                  id="Aspect"
                  type="number"
                  value={formData.Aspect}
                  onChange={(e) => setFormData({ ...formData, Aspect: e.target.value })}
                  placeholder="51"
                  required
                />
              </div>

              <div>
                <Label htmlFor="Slope">Slope (degrees)</Label>
                <Input
                  id="Slope"
                  type="number"
                  value={formData.Slope}
                  onChange={(e) => setFormData({ ...formData, Slope: e.target.value })}
                  placeholder="3"
                  required
                />
              </div>

              <div>
                <Label htmlFor="Horizontal_Distance_To_Hydrology">Distance to Hydrology (m)</Label>
                <Input
                  id="Horizontal_Distance_To_Hydrology"
                  type="number"
                  value={formData.Horizontal_Distance_To_Hydrology}
                  onChange={(e) => setFormData({ ...formData, Horizontal_Distance_To_Hydrology: e.target.value })}
                  placeholder="258"
                  required
                />
              </div>

              <div>
                <Label htmlFor="Vertical_Distance_To_Hydrology">Vertical Distance to Hydrology (m)</Label>
                <Input
                  id="Vertical_Distance_To_Hydrology"
                  type="number"
                  value={formData.Vertical_Distance_To_Hydrology}
                  onChange={(e) => setFormData({ ...formData, Vertical_Distance_To_Hydrology: e.target.value })}
                  placeholder="0"
                  required
                />
              </div>

              <div>
                <Label htmlFor="Horizontal_Distance_To_Roadways">Distance to Roadways (m)</Label>
                <Input
                  id="Horizontal_Distance_To_Roadways"
                  type="number"
                  value={formData.Horizontal_Distance_To_Roadways}
                  onChange={(e) => setFormData({ ...formData, Horizontal_Distance_To_Roadways: e.target.value })}
                  placeholder="510"
                  required
                />
              </div>

              <div>
                <Label htmlFor="Hillshade_9am">Hillshade 9am</Label>
                <Input
                  id="Hillshade_9am"
                  type="number"
                  value={formData.Hillshade_9am}
                  onChange={(e) => setFormData({ ...formData, Hillshade_9am: e.target.value })}
                  placeholder="221"
                  required
                />
              </div>

              <div>
                <Label htmlFor="Hillshade_Noon">Hillshade Noon</Label>
                <Input
                  id="Hillshade_Noon"
                  type="number"
                  value={formData.Hillshade_Noon}
                  onChange={(e) => setFormData({ ...formData, Hillshade_Noon: e.target.value })}
                  placeholder="232"
                  required
                />
              </div>

              <div>
                <Label htmlFor="Hillshade_3pm">Hillshade 3pm</Label>
                <Input
                  id="Hillshade_3pm"
                  type="number"
                  value={formData.Hillshade_3pm}
                  onChange={(e) => setFormData({ ...formData, Hillshade_3pm: e.target.value })}
                  placeholder="148"
                  required
                />
              </div>

              <div>
                <Label htmlFor="Horizontal_Distance_To_Fire_Points">Distance to Fire Points (m)</Label>
                <Input
                  id="Horizontal_Distance_To_Fire_Points"
                  type="number"
                  value={formData.Horizontal_Distance_To_Fire_Points}
                  onChange={(e) => setFormData({ ...formData, Horizontal_Distance_To_Fire_Points: e.target.value })}
                  placeholder="0"
                  required
                />
              </div>

              <div>
                <Label htmlFor="Wilderness_Area">Wilderness Area</Label>
                <Select value={formData.Wilderness_Area} onValueChange={(v) => setFormData({ ...formData, Wilderness_Area: v })}>
                  <SelectTrigger><SelectValue placeholder="Select wilderness area" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="rawah wilderness area">Rawah Wilderness Area</SelectItem>
                    <SelectItem value="neota wilderness area">Neota Wilderness Area</SelectItem>
                    <SelectItem value="comanche peak wilderness area">Comanche Peak Wilderness Area</SelectItem>
                    <SelectItem value="cache la poudre wilderness area">Cache La Poudre Wilderness Area</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label htmlFor="Soil_Type">Soil Type</Label>
                <Select value={formData.Soil_Type} onValueChange={(v) => setFormData({ ...formData, Soil_Type: v })}>
                  <SelectTrigger><SelectValue placeholder="Select soil type" /></SelectTrigger>
                  <SelectContent className="max-h-[300px]">
                    <SelectItem value="cathedral family - rock outcrop complex, extremely stony.">Cathedral Family - Rock Outcrop Complex</SelectItem>
                    <SelectItem value="vanet - ratake families complex, very stony.">Vanet - Ratake Families Complex</SelectItem>
                    <SelectItem value="haploborolis - rock outcrop complex, rubbly.">Haploborolis - Rock Outcrop Complex</SelectItem>
                    <SelectItem value="ratake family - rock outcrop complex, rubbly.">Ratake Family - Rock Outcrop Complex</SelectItem>
                    <SelectItem value="vanet family - rock outcrop complex complex, rubbly.">Vanet Family - Rock Outcrop Complex</SelectItem>
                    <SelectItem value="vanet - wetmore families - rock outcrop complex, stony.">Vanet - Wetmore Families Complex</SelectItem>
                    <SelectItem value="gothic family.">Gothic Family</SelectItem>
                    <SelectItem value="supervisor - limber families complex.">Supervisor - Limber Families Complex</SelectItem>
                    <SelectItem value="troutville family, very stony.">Troutville Family</SelectItem>
                    <SelectItem value="bullwark - catamount families - rock outcrop complex, rubbly.">Bullwark - Catamount Families Complex</SelectItem>
                    <SelectItem value="bullwark - catamount families - rock land complex, rubbly.">Bullwark - Catamount Families - Rock Land</SelectItem>
                    <SelectItem value="legault family - rock land complex, stony.">Legault Family - Rock Land Complex</SelectItem>
                    <SelectItem value="catamount family - rock land - bullwark family complex, rubbly.">Catamount Family - Rock Land - Bullwark</SelectItem>
                    <SelectItem value="pachic argiborolis - aquolis complex.">Pachic Argiborolis - Aquolis Complex</SelectItem>
                    <SelectItem value="unspecified in the usfs soil and elu survey.">Unspecified in USFS Survey</SelectItem>
                    <SelectItem value="cryaquolis - cryoborolis complex.">Cryaquolis - Cryoborolis Complex</SelectItem>
                    <SelectItem value="gateview family - cryaquolis complex.">Gateview Family - Cryaquolis Complex</SelectItem>
                    <SelectItem value="rogert family, very stony.">Rogert Family</SelectItem>
                    <SelectItem value="typic cryaquolis - borohemists complex.">Typic Cryaquolis - Borohemists Complex</SelectItem>
                    <SelectItem value="typic cryaquepts - typic cryaquolls complex.">Typic Cryaquepts - Typic Cryaquolls</SelectItem>
                    <SelectItem value="typic cryaquolls - leighcan family, till substratum complex.">Typic Cryaquolls - Leighcan Family</SelectItem>
                    <SelectItem value="leighcan family, till substratum, extremely bouldery.">Leighcan Family (Till Substratum)</SelectItem>
                    <SelectItem value="leighcan family, till substratum - typic cryaquolls complex.">Leighcan - Typic Cryaquolls Complex</SelectItem>
                    <SelectItem value="leighcan family, extremely stony.">Leighcan Family (Extremely Stony)</SelectItem>
                    <SelectItem value="leighcan family, warm, extremely stony.">Leighcan Family (Warm)</SelectItem>
                    <SelectItem value="granile - catamount families complex, very stony.">Granile - Catamount Families</SelectItem>
                    <SelectItem value="leighcan family, warm - rock outcrop complex, extremely stony.">Leighcan (Warm) - Rock Outcrop</SelectItem>
                    <SelectItem value="leighcan family - rock outcrop complex, extremely stony.">Leighcan - Rock Outcrop Complex</SelectItem>
                    <SelectItem value="como - legault families complex, extremely stony.">Como - Legault Families</SelectItem>
                    <SelectItem value="como family - rock land - legault family complex, extremely stony.">Como - Rock Land - Legault</SelectItem>
                    <SelectItem value="leighcan - catamount families complex, extremely stony.">Leighcan - Catamount Families</SelectItem>
                    <SelectItem value="catamount family - rock outcrop - leighcan family complex, extremely stony.">Catamount - Rock Outcrop - Leighcan</SelectItem>
                    <SelectItem value="leighcan - catamount families - rock outcrop complex, extremely stony.">Leighcan - Catamount - Rock Outcrop</SelectItem>
                    <SelectItem value="cryorthents - rock land complex, extremely stony.">Cryorthents - Rock Land Complex</SelectItem>
                    <SelectItem value="cryumbrepts - rock outcrop - cryaquepts complex.">Cryumbrepts - Rock Outcrop Complex</SelectItem>
                    <SelectItem value="bross family - rock land - cryumbrepts complex, extremely stony.">Bross Family - Rock Land - Cryumbrepts</SelectItem>
                    <SelectItem value="rock outcrop - cryumbrepts - cryorthents complex, extremely stony.">Rock Outcrop - Cryumbrepts - Cryorthents</SelectItem>
                    <SelectItem value="leighcan - moran families - cryaquolls complex, extremely stony.">Leighcan - Moran - Cryaquolls</SelectItem>
                    <SelectItem value="moran family - cryorthents - leighcan family complex, extremely stony.">Moran - Cryorthents - Leighcan</SelectItem>
                    <SelectItem value="moran family - cryorthents - rock land complex, extremely stony.">Moran - Cryorthents - Rock Land</SelectItem>
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

export default PredictForest;
