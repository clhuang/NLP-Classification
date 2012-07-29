package argumentClassification;

import java.util.ArrayDeque;
import java.util.Deque;

public class SentenceUtils {
	public static ArgumentClassifierToken getCommonAncestor(ArgumentClassifierToken a, ArgumentClassifierToken b){
		
		if (!a.getSentenceTokens().equals(b.getSentenceTokens()))
			return null;

		Deque<Integer> aAncestorIndices = new ArrayDeque<Integer>();
		Deque<Integer> bAncestorIndices = new ArrayDeque<Integer>();
		ArgumentClassifierToken possibleAncestor = a;

		aAncestorIndices.addFirst(a.sentenceIndex);
		while (possibleAncestor.parentIndex >= 0){
			possibleAncestor = (ArgumentClassifierToken) possibleAncestor.getParent();
			aAncestorIndices.addFirst(possibleAncestor.sentenceIndex);
		}

		possibleAncestor = b;
		bAncestorIndices.addFirst(b.sentenceIndex);
		while (possibleAncestor.parentIndex >= 0){
			possibleAncestor = (ArgumentClassifierToken) possibleAncestor.getParent();
			bAncestorIndices.addFirst(possibleAncestor.sentenceIndex);
		}

		int ancestorIndex = 0;
		while (!aAncestorIndices.isEmpty() && !bAncestorIndices.isEmpty() && aAncestorIndices.peekFirst().equals(bAncestorIndices.peekFirst())){
			ancestorIndex = aAncestorIndices.removeFirst();
			bAncestorIndices.removeFirst();
		}

		return a.getSentenceTokens().get(ancestorIndex);

	}

	public static int ancestorPathLength(ArgumentClassifierToken a, ArgumentClassifierToken ancestor){
		if (!a.getSentenceTokens().equals(ancestor.getSentenceTokens()))
			return -1;

		return ancestorPath(a, ancestor).size() - 1;
	}

	public static Deque<ArgumentClassifierToken> ancestorPath(ArgumentClassifierToken a,
			ArgumentClassifierToken ancestor) {
		if (!a.getSentenceTokens().equals(ancestor.getSentenceTokens()))
			return null;

		Deque<ArgumentClassifierToken> path = new ArrayDeque<ArgumentClassifierToken>();
		ArgumentClassifierToken currentToken = a;
		path.add(a);

		while (!currentToken.equals(ancestor)){
			currentToken = (ArgumentClassifierToken) currentToken.getParent();
			if (currentToken == null)
				return null;
			path.add(currentToken);
		}

		return path;

	}

	public static int dependencyPathLength(ArgumentClassifierToken a,
			ArgumentClassifierToken b){

		if (!a.getSentenceTokens().equals(b.getSentenceTokens()))
			return -1;

		ArgumentClassifierToken ancestor = getCommonAncestor(a, b);

		Deque<ArgumentClassifierToken> pathA = ancestorPath(a, ancestor);
		Deque<ArgumentClassifierToken> pathB = ancestorPath(b, ancestor);
		
		return pathA.size() + pathB.size() - 2;

	}

	public static Deque<ArgumentClassifierToken> dependencyPath(ArgumentClassifierToken a,
			ArgumentClassifierToken b){

		if (!a.getSentenceTokens().equals(b.getSentenceTokens()))
			return null;

		ArgumentClassifierToken ancestor = getCommonAncestor(a, b);

		Deque<ArgumentClassifierToken> pathA = ancestorPath(a, ancestor);
		Deque<ArgumentClassifierToken> pathB = ancestorPath(b, ancestor);

		pathB.removeLast(); //ancestor is last thing in both pathA and pathB
		while (!pathB.isEmpty())
			pathA.addLast(pathB.removeLast());

		return pathA;
	}
}
