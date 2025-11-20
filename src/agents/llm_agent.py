# src/agents/llm_agent.py
from openai import OpenAI, RateLimitError
from typing import List, Dict, Union, Optional
from src.config.settings import Settings
import json
import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class ChatAgent:
    """OpenAI Chatbot client."""

    def __init__(self, config: Settings):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self.model = config.openai_llm_model

    def chat(self, messages: List[dict]) -> str:
        """
        Send a list of messages to the OpenAI chat model and get the response.
        Uses exponential backoff retry logic for rate limit errors.

        Args:
            messages: List of message dicts (e.g., [{"role": "user", "content": "Hello"}])

        Returns:
            The assistant's reply as a string.
        """
        max_retries = 5
        base_delay = 1.0  # Start with 1 second delay
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
                return response.choices[0].message.content
            except RateLimitError as e:
                if attempt == max_retries - 1:
                    # Last attempt, raise the error
                    raise
                
                # Calculate exponential backoff delay
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Rate limit hit on chat completion. Retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            except Exception as e:
                # For other errors, raise immediately
                raise

    def chat_single(self, prompt: str) -> str:
        """
        Send a single prompt to the OpenAI chat model and get the response.

        Args:
            prompt: The user's prompt as a string.

        Returns:
            The assistant's reply as a string.
        """
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages)

    def label_cluster(self, feedback_texts: List[str]) -> str:
        """
        Generate a descriptive label for a cluster based on sample feedback texts.
        
        Args:
            feedback_texts: Sample customer feedback from the cluster
            
        Returns:
            A concise theme/label for the cluster
        """
        prompt = f"""You are analyzing customer feedback for Vessi waterproof shoes. Below are {len(feedback_texts)} sample comments from a cluster of similar feedback.

            Generate a concise, descriptive label (2-5 words) that captures the main theme or issue.

            Sample feedback:
            {chr(10).join(f"{i+1}. {text[:200]}" for i, text in enumerate(feedback_texts))}

            Return only the label, no explanation:"""
        
        return self.chat_single(prompt).strip()

    def tag_feedback(self, feedback_text: str, categories: List[str], allow_multiple: bool = True) -> List[str]:
        """
        Tag a single piece of customer feedback with predefined categories.
        
        Args:
            feedback_text: Customer feedback text to tag
            categories: List of possible category tags
            allow_multiple: If True, can return multiple tags; if False, returns only the best single tag
            
        Returns:
            List of category strings (single item list if allow_multiple=False)
        """
        categories_list = "\n".join(f"- {cat}" for cat in categories)
        
        if allow_multiple:
            prompt = f"""You are analyzing customer feedback for Vessi waterproof shoes. 

            Customer feedback: "{feedback_text}"

            Available categories:
            {categories_list}

            Task: Identify ALL relevant categories that apply to this feedback. Be thorough - a single piece of feedback can have multiple issues.

            IMPORTANT: 
            - Look for every distinct issue or concern mentioned
            - A review mentioning "leaking AND sizing" should get BOTH tags
            - Return ALL applicable categories, not just the primary one
            - If no categories apply, return an empty array

            Return ONLY a valid JSON array of category names that match EXACTLY as written above.

            Example responses:
            ["Waterproof Leak"]
            ["Waterproof Leak", "Sizes not standard"]
            []

            Response:"""
        else:
            prompt = f"""You are analyzing customer feedback for Vessi waterproof shoes. 

            Customer feedback: "{feedback_text}"

            Available categories:
            {categories_list}

            Task: Identify the SINGLE most important category that best describes the primary issue in this feedback.

            Return ONLY the category name exactly as written above, with no additional text or explanation.

            Response:"""
        
        response = self.chat_single(prompt).strip()
        
        if allow_multiple:
            # Try multiple parsing strategies
            tags = self._parse_json_array(response, categories)
            return tags if tags else ["Uncategorized"]
        else:
            # Single tag mode - validate and return as list
            tag = self._parse_single_tag(response, categories)
            return [tag]

    def _parse_json_array(self, response: str, categories: List[str]) -> List[str]:
        """Parse LLM response as JSON array with multiple fallback strategies."""
        # Strategy 1: Direct JSON parsing
        try:
            # Remove markdown code blocks if present
            response = re.sub(r'```json\s*|\s*```', '', response)
            tags = json.loads(response)
            if isinstance(tags, list):
                valid_tags = [tag for tag in tags if tag in categories]
                return valid_tags
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract array from text
        try:
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                tags = json.loads(match.group(0))
                if isinstance(tags, list):
                    valid_tags = [tag for tag in tags if tag in categories]
                    if valid_tags:
                        return valid_tags
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Strategy 3: Find category names in response
        response_lower = response.lower()
        found_tags = [cat for cat in categories if cat.lower() in response_lower]
        if found_tags:
            return found_tags
        
        # Strategy 4: Check if response is a single category
        if response in categories:
            return [response]
        
        for cat in categories:
            if cat.lower() == response.lower().strip('"\''):
                return [cat]
        
        return []

    def _parse_single_tag(self, response: str, categories: List[str]) -> str:
        """Parse LLM response as single tag."""
        # Clean response
        response = response.strip('"\'')
        
        # Direct match
        if response in categories:
            return response
        
        # Case-insensitive match
        for cat in categories:
            if cat.lower() == response.lower():
                return cat
        
        # Partial match
        response_lower = response.lower()
        for cat in categories:
            if cat.lower() in response_lower or response_lower in cat.lower():
                return cat
        
        return "Uncategorized"

    def tag_feedback_batch(self, feedback_texts: List[str], categories: List[str], 
                          allow_multiple: bool = True) -> List[List[str]]:
        """
        Tag multiple pieces of feedback in a single API call (more efficient).
        
        Args:
            feedback_texts: List of customer feedback texts to tag
            categories: List of possible category tags
            allow_multiple: If True, can return multiple tags per feedback
            
        Returns:
            List of tag lists (each feedback gets a list of tags)
        """
        categories_list = "\n".join(f"- {cat}" for cat in categories)
        feedback_list = "\n".join(f"{i+1}. {text}" for i, text in enumerate(feedback_texts))
        
        if allow_multiple:
            prompt = f"""You are analyzing customer feedback for Vessi waterproof shoes.

                    Available categories:
                    {categories_list}

                    Customer feedback to tag:
                    {feedback_list}

                    Task: For each piece of feedback, identify ALL relevant categories. Be thorough - look for every distinct issue mentioned.

                    IMPORTANT:
                    - A review can have multiple issues (e.g., "leaking AND uncomfortable")
                    - Return ALL applicable categories for each feedback item
                    - If no categories apply to a specific feedback, use an empty array

                    Return ONLY a valid JSON object where keys are the feedback numbers (1, 2, 3, etc.) and values are arrays of matching category names.

                    Example format:
                    {{
                    "1": ["Waterproof Leak", "Sizes not standard"],
                    "2": ["Toe Area too narrow"],
                    "3": []
                    }}

                    Response:"""
        else:
            prompt = f"""You are analyzing customer feedback for Vessi waterproof shoes.

            Available categories:
            {categories_list}

            Customer feedback to tag:
            {feedback_list}

            Task: For each piece of feedback, identify the SINGLE most important category.

            Return ONLY a valid JSON object where keys are the feedback numbers (1, 2, 3, etc.) and values are the category names.

            Example format:
            {{
            "1": "Waterproof Leak",
            "2": "Toe Area too narrow",
            "3": "Uncategorized"
            }}

            Response:"""
        
        response = self.chat_single(prompt).strip()
        
        try:
            # Remove markdown code blocks if present
            response = re.sub(r'```json\s*|\s*```', '', response)
            result_dict = json.loads(response)
            results = []
            
            for i in range(len(feedback_texts)):
                key = str(i + 1)
                if key in result_dict:
                    value = result_dict[key]
                    if allow_multiple:
                        # Ensure value is a list
                        if isinstance(value, list):
                            valid_tags = [tag for tag in value if tag in categories]
                            results.append(valid_tags if valid_tags else ["Uncategorized"])
                        elif isinstance(value, str) and value in categories:
                            # Handle case where LLM returned single string instead of array
                            results.append([value])
                        else:
                            results.append(["Uncategorized"])
                    else:
                        # Single tag mode - return as list
                        if isinstance(value, str) and value in categories:
                            results.append([value])
                        elif isinstance(value, list) and value and value[0] in categories:
                            results.append([value[0]])
                        else:
                            results.append(["Uncategorized"])
                else:
                    results.append(["Uncategorized"])
            
            return results
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback to individual tagging
            print(f"Batch parsing failed: {e}. Falling back to individual tagging.")
            return [self.tag_feedback(text, categories, allow_multiple) for text in feedback_texts]


class FeedbackTagger:
    """Tag customer feedback with predefined issue categories."""
    
    # Predefined Vessi product issue categories
    DEFAULT_CATEGORIES = [
        "Waterproof Leak",
        "Upper Knit Separation",
        "Insole Issue",
        "Inner Lining Rip",
        "Glue Gap",
        "Discolouration",
        "Sizes not standard",
        "Toe Area too narrow",
        "Toe area too big",
        "Instep too small",
        "instep too high",
        "shoe too narrow",
        "shoe too wide",
        "half size requests",
        "no heel lock/heel slip",
        "Lack of grip/traction",
        "Squeaky sound",
        "Not breathable enough",
        "hard to take off",
        "hard to put on",
        "Lack of support",
        "Heel Cup - too big",
        "Smelly",
        "Back Heel Rubbing",
        "Warping",
        "Stains",
        "Looks different than picture/ ugly/ not what was expected",
        "blisters",
        "Too Bulky",
        "Too Heavy"
    ]
    
    def __init__(self, config: Settings, custom_categories: Optional[List[str]] = None):
        """
        Initialize the feedback tagger.
        
        Args:
            config: Settings object with OpenAI configuration
            custom_categories: Optional custom category list (uses DEFAULT_CATEGORIES if None)
        """
        self.agent = ChatAgent(config)
        self.categories = custom_categories if custom_categories else self.DEFAULT_CATEGORIES
        self.max_workers = config.max_workers
    
    def tag_single(self, feedback_text: str, allow_multiple: bool = True) -> List[str]:
        """
        Tag a single piece of customer feedback.
        
        Args:
            feedback_text: Customer feedback text
            allow_multiple: If True, returns list of all applicable tags; if False, returns single-item list with best tag
            
        Returns:
            List of category strings
        """
        return self.agent.tag_feedback(feedback_text, self.categories, allow_multiple)
    
    def tag_batch(self, feedback_texts: List[str], allow_multiple: bool = True, 
                  batch_size: int = 10) -> List[List[str]]:
        """
        Tag multiple pieces of feedback efficiently using multithreading.
        
        Args:
            feedback_texts: List of customer feedback texts
            allow_multiple: If True, returns lists of multiple tags; if False, returns single-tag lists
            batch_size: Number of feedback items to process per API call (max 20 recommended)
            
        Returns:
            List of tag lists (each feedback item gets a list of tags)
        """
        # Split feedback into batches
        batches = []
        for i in range(0, len(feedback_texts), batch_size):
            batch = feedback_texts[i:i + batch_size]
            batches.append((i, batch))
        
        # Process batches in parallel using ThreadPoolExecutor
        results_dict = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batch processing tasks
            future_to_batch = {
                executor.submit(
                    self.agent.tag_feedback_batch, 
                    batch, 
                    self.categories, 
                    allow_multiple
                ): (batch_idx, batch)
                for batch_idx, batch in batches
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_idx, batch = future_to_batch[future]
                try:
                    batch_results = future.result()
                    results_dict[batch_idx] = batch_results
                except Exception as e:
                    # Handle errors gracefully - return "Uncategorized" for failed batches
                    print(f"Error processing batch starting at index {batch_idx}: {e}")
                    results_dict[batch_idx] = [["Uncategorized"] for _ in batch]
        
        # Reconstruct results in original order
        all_results = []
        for batch_idx in sorted(results_dict.keys()):
            all_results.extend(results_dict[batch_idx])
        
        return all_results
    
    
    def get_category_distribution(self, feedback_texts: List[str], 
                                 allow_multiple: bool = True) -> Dict[str, int]:
        """
        Get distribution of categories across multiple feedback items.
        
        Args:
            feedback_texts: List of customer feedback texts
            allow_multiple: If True, counts all tags per feedback; if False, counts only primary tag
            
        Returns:
            Dict mapping category names to counts
        """
        all_tags = self.tag_batch(feedback_texts, allow_multiple)
        
        distribution = {cat: 0 for cat in self.categories}
        distribution["Uncategorized"] = 0
        
        for tag_list in all_tags:
            for tag in tag_list:
                if tag in distribution:
                    distribution[tag] += 1
                else:
                    distribution["Uncategorized"] += 1
        
        # Remove categories with zero counts and sort by count descending
        return dict(sorted(
            {k: v for k, v in distribution.items() if v > 0}.items(),
            key=lambda x: x[1],
            reverse=True
        ))
    
    def get_multi_tag_stats(self, feedback_texts: List[str]) -> Dict[str, any]:
        """
        Get statistics about multi-tag occurrences.
        
        Args:
            feedback_texts: List of customer feedback texts
            
        Returns:
            Dict with statistics about tagging patterns
        """
        all_tags = self.tag_batch(feedback_texts, allow_multiple=True)
        
        tag_counts = [len(tags) for tags in all_tags]
        
        return {
            'total_feedback': len(feedback_texts),
            'avg_tags_per_feedback': sum(tag_counts) / len(tag_counts) if tag_counts else 0,
            'max_tags_single_feedback': max(tag_counts) if tag_counts else 0,
            'single_tag_count': sum(1 for c in tag_counts if c == 1),
            'multi_tag_count': sum(1 for c in tag_counts if c > 1),
            'no_tag_count': sum(1 for tags in all_tags if tags == ["Uncategorized"]),
            'distribution': self.get_category_distribution(feedback_texts, allow_multiple=True)
        }
    
    def add_category(self, category: str):
        """Add a new category to the tagger."""
        if category not in self.categories:
            self.categories.append(category)
    
    def remove_category(self, category: str):
        """Remove a category from the tagger."""
        if category in self.categories:
            self.categories.remove(category)
    
    def get_categories(self) -> List[str]:
        """Get current list of categories."""
        return self.categories.copy()