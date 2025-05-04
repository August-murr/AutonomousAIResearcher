import modal
from concurrent.futures import ThreadPoolExecutor

class ModalCodeExecuter:
    """
    Executes Python code within a Modal Sandbox environment,
    handling concurrent execution of multiple code strings using ThreadPoolExecutor.
    Example usage:
        image = (modal.Image.debian_slim(python_version="3.12").pip_install("torch"))
        app = modal.App.lookup("my-app", create_if_missing=True)
        executer = ModalCodeExecuter(image=image, app=app,gpu="T4", max_workers=16)
        result = executer.execute([code_string1, code_string2,...]) # List of code strings
    """
    def __init__(self, image, app, gpu=None, max_workers=32):
        """
        Initializes the ModalCodeExecuter.

        Args:
            image: The modal.Image to use for the sandbox.
            app: The modal.App associated with the sandbox.
            gpu: The GPU configuration for the sandbox (default: None).
            max_workers: The maximum number of threads (and thus concurrent sandboxes)
                         to use (default: 10). Adjust based on expected load and
                         Modal plan limits.
        """
        self.image = image
        self.app = app
        self.gpu = gpu
        self.max_workers = max_workers

    def _execute_single_sync(self, code_string: str):
        """
        Executes a single Python code string in a dedicated Modal Sandbox (Synchronous).

        Args:
            code_string: The Python code to execute.

        Returns:
            The raw standard output bytes if not empty, otherwise the
            raw standard error bytes. Returns empty bytes if both are empty,
            or an Exception object on failure.
        """
        sb = None
        # print(f"Thread starting execution for: {code_string[:20]}...") # Debug print
        try:
            # Create sandbox - This is a blocking call
            sb = modal.Sandbox.create(
                app=self.app,
                image=self.image,
                gpu=self.gpu,
                timeout=60 # Added a timeout for safety
            )

            # Execute code - This is a blocking call until sb.exec returns
            p = sb.exec("python", "-c", code_string)

            # Wait for the process to complete - This is a blocking call
            p.wait()

            # Read stdout first
            result_bytes = p.stdout.read()

            if result_bytes:
                # If stdout has content, return the raw bytes
                # print(f"Thread finished (stdout): {code_string[:20]}...") # Debug print
                return result_bytes
            else:
                # If stdout is empty, read and return raw stderr bytes
                error_bytes = p.stderr.read()
                # print(f"Thread finished (stderr/empty): {code_string[:20]}...") # Debug print
                return error_bytes # Return stderr bytes (could be empty bytes b'')

        except Exception as e:
            # Catch potential exceptions during sandbox creation or execution
            # print(f"Thread finished (exception): {code_string[:20]}... {e}") # Debug print
            return e # Return the exception itself
        finally:
            # Ensure sandbox is terminated even if errors occur
            if sb:
                # print(f"Terminating sandbox for: {code_string[:20]}...") # Debug print
                sb.terminate() # Blocking call


    def execute(self, code_strings: list[str]):
        """
        Creates sandboxes, executes the provided Python code strings concurrently
        using a ThreadPoolExecutor, terminates the sandboxes, and returns a list
        of outputs (bytes) or errors.

        Args:
            code_strings: A list of Python code strings to execute.

        Returns:
            A list containing the raw standard output/error bytes or Exception objects
            for each corresponding code string.
        """
        results = []
        # Use ThreadPoolExecutor to manage concurrent threads
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # map submits tasks and returns results in the order of the input iterable
            # It waits for all tasks to complete.
            future_results = executor.map(self._execute_single_sync, code_strings)
            # Convert map iterator to list to ensure all tasks are finished and get results
            results = list(future_results)

        return results