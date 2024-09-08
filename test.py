import unittest
import unittest.test
from parcial import criterio, indicadora, label_tg,label_tg_inverse, prueba_kr, Mt_flow 
import numpy as np

class prueba(unittest.TestCase):
    def test_model(self):
        model = Mt_flow()
        salida = model.final(1,2)
        
        if salida["success"]==True:
            self.assertTrue(salida["success"],"Completed succesfully ...")
            self.assertGreaterEqual(salida["accuracy"],70,'Excellent performance')
            a = "Completed succesfully ..."
            b = salida["accuracy"]
            return {'Procces':a,'Accuracy':b}
        else:
            a = "Not completed succesfully ..."
            b = salida["message"]
            return {'Procces':a,',Message':b}

pb = prueba()
print(pb.test_model())

if __name__ == "__main__":
    unittest.main()