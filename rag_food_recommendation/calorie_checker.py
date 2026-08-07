from shared_functions import *

# Global variables
food_items = []
calorie_budget = None


def main():
    """Main function for the interactive calorie budget food checker"""
    try:
        print("🍎 Food Calorie Checker")
        print("=" * 50)
        print("Loading food database...")

        # Load food data from file
        global food_items
        food_items = load_food_data('./FoodDataSet.json')
        print(f"✅ Loaded {len(food_items)} food items successfully")

        # Create and populate search collection
        collection = create_similarity_search_collection(
            "calorie_checker_search",
            {'description': 'A collection for calorie-budget food search'}
        )
        populate_similarity_collection(collection, food_items)

        # Ask the user for their calorie budget before starting
        global calorie_budget
        calorie_budget = get_calorie_budget()

        # Start interactive chatbot
        calorie_checker_chatbot(collection)

    except Exception as error:
        print(f"❌ Error initializing system: {error}")


def get_calorie_budget() -> int:
    """Prompt the user for a calorie budget, validating the input"""
    while True:
        raw_input_value = input("\n🎯 What's your calorie budget per serving? ").strip()
        try:
            budget = int(raw_input_value)
        except ValueError:
            print("   Please enter a valid whole number (e.g. 400).")
            continue

        if budget <= 0:
            print("   Please enter a positive number.")
            continue

        return budget


def calorie_checker_chatbot(collection):
    """Interactive CLI chatbot for calorie-budget food search"""
    global calorie_budget

    print("\n" + "=" * 50)
    print("🤖 CALORIE BUDGET FOOD CHECKER")
    print("=" * 50)
    print(f"💰 Current calorie budget: {calorie_budget} calories per serving")
    print("Commands:")
    print("  • Type any food name or description to search")
    print("  • 'budget' - Change your calorie budget")
    print("  • 'help' - Show available commands")
    print("  • 'quit' or 'exit' - Exit the system")
    print("-" * 50)

    while True:
        try:
            # Get user input
            user_input = input("\n🔍 Search for food: ").strip()

            # Handle empty input
            if not user_input:
                print("   Please enter a search term or 'help' for commands")
                continue

            # Handle exit commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for using the Food Calorie Checker! Goodbye!")
                break

            # Handle help command
            elif user_input.lower() in ['help', 'h']:
                show_help_menu()

            # Handle budget change
            elif user_input.lower() == 'budget':
                calorie_budget = get_calorie_budget()
                print(f"✅ Budget updated to {calorie_budget} calories per serving")

            # Handle food search
            else:
                handle_calorie_search(collection, user_input)

        except KeyboardInterrupt:
            print("\n\n👋 System interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error processing request: {e}")


def show_help_menu():
    """Display help information for users"""
    print("\n📖 HELP MENU")
    print("-" * 30)
    print("Search Examples:")
    print("  • 'chocolate dessert' - Find chocolate desserts")
    print("  • 'Italian food' - Find Italian cuisine")
    print("  • 'grilled chicken' - Find grilled dishes")
    print("  • 'light breakfast' - Find lighter options")
    print("\nCommands:")
    print("  • 'budget' - Change your calorie budget")
    print("  • 'help' - Show this help menu")
    print("  • 'quit' - Exit the system")


def handle_calorie_search(collection, query: str):
    """Search for foods matching the query that also fit the calorie budget"""
    print(f"\n🔍 Searching for '{query}' within {calorie_budget} calories...")
    print("   Please wait...")

    # Perform similarity search filtered by the calorie budget
    results = perform_filtered_similarity_search(
        collection, query, max_calories=calorie_budget, n_results=5
    )

    if not results:
        print(f"❌ No foods matching '{query}' found within your {calorie_budget} calorie budget.")
        print("💡 Try a different search term, or raise your budget with 'budget'.")
        return

    print(f"\n✅ Found {len(results)} budget-friendly matches:")
    print("=" * 60)

    for i, result in enumerate(results, 1):
        percentage_score = result['similarity_score'] * 100
        calories = result['food_calories_per_serving']
        remaining = calorie_budget - calories

        print(f"\n{i}. 🍽️  {result['food_name']}")
        print(f"   📊 Match Score: {percentage_score:.1f}%")
        print(f"   🏷️  Cuisine: {result['cuisine_type']}")
        print(f"   🔥 Calories: {calories} / {calorie_budget} budget ({remaining} remaining)")
        print(f"   📝 Description: {result['food_description']}")

        # Add visual separator
        if i < len(results):
            print("   " + "-" * 50)

    print("=" * 60)


if __name__ == "__main__":
    main()
