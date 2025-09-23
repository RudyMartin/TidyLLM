"""
Data Normalization Utilities
============================

Robust normalization functions for handling various data formats
returned by different TidyLLM managers and services.
"""

from typing import Dict, List, Any, Union, Optional


class DataNormalizer:
    """Comprehensive data normalization utility for TidyLLM portals."""

    @staticmethod
    def ensure_list(data: Any, default: Optional[List[str]] = None) -> List[str]:
        """
        Ensure data is a list of strings, with robust normalization.

        Handles:
        - List[str]: Pass through as-is
        - str: Convert to single-item list
        - None/Empty: Return default or empty list
        - Other types: Convert to string and wrap in list

        Args:
            data: Input data in various formats
            default: Default list to return if data is None/empty

        Returns:
            List[str]: Normalized list of strings
        """
        if default is None:
            default = []

        # Handle None or empty
        if data is None or data == "":
            return default

        # Already a list
        if isinstance(data, list):
            # Ensure all items are strings
            return [str(item) for item in data if item is not None]

        # Single string
        if isinstance(data, str):
            return [data.strip()] if data.strip() else default

        # Other types - convert to string
        return [str(data)]

    @staticmethod
    def normalize_to_status_dict(data: Any) -> Dict[str, bool]:
        """
        Normalize various data formats to a simple {name: bool} status dictionary.

        Handles:
        - Dict[str, bool]: Pass through as-is
        - Dict[str, Any]: Extract boolean status from complex values
        - List[Dict]: Extract status from list of system info dicts
        - List[str]: Convert to {item: True} for each item
        - None/Empty: Return empty dict

        Args:
            data: Input data in various formats

        Returns:
            Dict[str, bool]: Normalized status dictionary
        """
        # Handle None or empty
        if data is None:
            return {}

        # Already in correct format
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                # Handle different value types
                if isinstance(value, bool):
                    result[key] = value
                elif isinstance(value, dict):
                    # Extract 'available' or 'enabled' or 'active' field
                    result[key] = (
                        value.get('available',
                        value.get('enabled',
                        value.get('active',
                        value.get('initialized', False))))
                    )
                elif value is None:
                    result[key] = False
                else:
                    # Truthy check for other types
                    result[key] = bool(value)
            return result

        # List of dictionaries (like RAG system info)
        if isinstance(data, list):
            if not data:  # Empty list
                return {}

            # Check first item to determine list type
            first_item = data[0]

            if isinstance(first_item, dict):
                # List of system/service info dicts
                result = {}
                for i, item in enumerate(data):
                    # Try different key fields for the name
                    key = (
                        item.get('system_type') or
                        item.get('system_name') or
                        item.get('name') or
                        item.get('id') or
                        item.get('type') or
                        f'system_{i}'
                    )

                    # Try different status fields
                    status = (
                        item.get('available') if 'available' in item else
                        item.get('enabled') if 'enabled' in item else
                        item.get('active') if 'active' in item else
                        item.get('initialized') if 'initialized' in item else
                        item.get('status') == 'active' if 'status' in item else
                        True  # Default to True if no status field
                    )

                    result[key] = status
                return result

            elif isinstance(first_item, str):
                # Simple list of names - assume all are available
                return {item: True for item in data if item is not None}

            elif isinstance(first_item, tuple) and len(first_item) == 2:
                # List of (name, status) tuples
                return {name: bool(status) for name, status in data if name is not None}

            else:
                # Unknown list format - convert items to strings
                return {str(item): True for item in data if item is not None}

        # Fallback for other types
        return {}

    @staticmethod
    def extract_system_details(data: Any) -> Dict[str, Dict[str, Any]]:
        """
        Extract detailed system information while preserving all metadata.

        Args:
            data: Input data in various formats

        Returns:
            Dict[str, Dict]: Dictionary with full system details
        """
        if data is None:
            return {}

        # If already a dict of dicts, return as-is
        if isinstance(data, dict):
            if data and isinstance(next(iter(data.values())), dict):
                return data
            # Convert simple dict to detailed format
            return {
                key: {'available': bool(value), 'name': key}
                for key, value in data.items()
            }

        # List of system info dicts
        if isinstance(data, list):
            result = {}
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    key = (
                        item.get('system_type') or
                        item.get('name') or
                        f'system_{i}'
                    )
                    result[key] = item
                else:
                    # Convert simple items
                    key = str(item)
                    result[key] = {'name': key, 'available': True}
            return result

        return {}

    @staticmethod
    def get_availability_counts(data: Any) -> tuple[int, int]:
        """
        Get counts of available vs total systems.

        Args:
            data: Input data in various formats

        Returns:
            tuple: (available_count, total_count)
        """
        normalized = DataNormalizer.normalize_to_status_dict(data)
        total = len(normalized)
        available = sum(1 for status in normalized.values() if status)
        return available, total

    @staticmethod
    def normalize_bedrock_response(response: Any) -> Dict[str, Any]:
        """
        Normalize various Bedrock response formats to a standard structure.

        Handles responses from:
        - boto3 client (AWS SDK): {'body': <StreamingBody>}
        - infra_delegate: raw string or dict
        - mock clients: various formats
        - wrapper clients: custom response objects

        Args:
            response: Response in various formats

        Returns:
            Dict with standardized format: {'body': <readable_object>}
            Where readable_object has .read() method returning JSON bytes
        """
        import json
        import io

        # Case 1: Already boto3-style response with readable body
        if isinstance(response, dict) and 'body' in response:
            body = response['body']
            if hasattr(body, 'read'):
                # Already correct format
                return response
            elif isinstance(body, (str, bytes)):
                # String/bytes body - make it readable
                if isinstance(body, str):
                    body = body.encode()
                return {'body': io.BytesIO(body)}
            else:
                # Other body format - convert to JSON
                json_str = json.dumps(body)
                return {'body': io.BytesIO(json_str.encode())}

        # Case 2: Direct string response (from infra_delegate)
        elif isinstance(response, str):
            # Wrap in Claude response format
            content = {'content': [{'text': response}]}
            json_str = json.dumps(content)
            return {'body': io.BytesIO(json_str.encode())}

        # Case 3: Dict response (from some services)
        elif isinstance(response, dict):
            # Check if it's already in Claude format
            if 'content' in response:
                json_str = json.dumps(response)
                return {'body': io.BytesIO(json_str.encode())}
            elif 'text' in response:
                # Single text response
                content = {'content': [{'text': response['text']}]}
                json_str = json.dumps(content)
                return {'body': io.BytesIO(json_str.encode())}
            else:
                # Generic dict - treat as complete response
                json_str = json.dumps(response)
                return {'body': io.BytesIO(json_str.encode())}

        # Case 4: Response object with attributes
        elif hasattr(response, '__dict__'):
            # Extract response content
            if hasattr(response, 'content'):
                content = {'content': [{'text': str(response.content)}]}
            elif hasattr(response, 'text'):
                content = {'content': [{'text': str(response.text)}]}
            else:
                # Convert entire object to dict
                content = {'content': [{'text': str(response)}]}

            json_str = json.dumps(content)
            return {'body': io.BytesIO(json_str.encode())}

        # Case 5: Fallback for unknown types
        else:
            content = {'content': [{'text': str(response)}]}
            json_str = json.dumps(content)
            return {'body': io.BytesIO(json_str.encode())}


# Convenience functions
def normalize_status(data: Any) -> Dict[str, bool]:
    """Convenience function to normalize data to status dict."""
    return DataNormalizer.normalize_to_status_dict(data)


def get_system_counts(data: Any) -> tuple[int, int]:
    """Convenience function to get availability counts."""
    return DataNormalizer.get_availability_counts(data)


# Test the normalizer with various formats
if __name__ == "__main__":
    # Test cases
    test_cases = [
        # Simple dict
        {"system1": True, "system2": False},

        # Complex dict
        {"rag1": {"available": True}, "rag2": {"enabled": False}},

        # List of system info dicts (like RAG systems)
        [
            {"system_type": "ai_powered", "available": True},
            {"name": "postgres", "available": False},
        ],

        # Simple list
        ["system1", "system2", "system3"],

        # Empty/None
        None,
        [],
        {},
    ]

    normalizer = DataNormalizer()
    for i, test_data in enumerate(test_cases):
        print(f"\nTest {i+1}: {type(test_data).__name__}")
        print(f"Input: {test_data}")
        result = normalizer.normalize_to_status_dict(test_data)
        print(f"Output: {result}")
        counts = normalizer.get_availability_counts(test_data)
        print(f"Counts: {counts[0]}/{counts[1]} available")