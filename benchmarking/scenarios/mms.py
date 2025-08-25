import os

from common.helpers import read_test_data
from locust import HttpUser, between, task

ModelCount = 10  # use --iterations 10

TestData = {
    "iris": read_test_data("iris"),
    "summodel": read_test_data("summodel"),
}

class MMSUser(HttpUser):
    wait_time = between(0.0, 0.0)
    host = f"http://{os.environ.get('MLSERVER_HOST','127.0.0.1')}:{os.environ.get('MLSERVER_HTTP_PORT','8080')}"

    @task
    def cycle_models(self):
        # Select which base to run from env via LOCUST_TASK_SET? Keep it simple: run both sequentially.
        for base in ("summodel", "iris"):
            model_name = f"{base}-{self.environment.runner.iteration + 1 if self.environment.runner else 1}"
            d = TestData[base.replace("-", "_")]
            payload = d["rest"]
            self.client.post(f"/v2/repository/models/{model_name}/load", name=f"Load {model_name}")
            self.client.post(f"/v2/models/{model_name}/infer", json=payload, name=f"Infer {model_name}")
            # Not unloading here (to mirror your commented-out unload)
