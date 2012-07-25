package argumentClassification;

import java.util.ArrayDeque;
import java.util.Deque;

import util.Token;

public class SentenceUtils {
	public static Token getCommonAncestor(Token a, Token b){

		if (a.getSentenceTokens() != b.getSentenceTokens())
			return null;

		Deque<Integer> aAncestorIndices = new ArrayDeque<Integer>();
		Deque<Integer> bAncestorIndices = new ArrayDeque<Integer>();
		Token possibleAncestor = a;

		while(possibleAncestor.parentIndex >= 0){
			aAncestorIndices.addFirst(possibleAncestor.sentenceIndex);
			possibleAncestor = possibleAncestor.getParent();
		}

		possibleAncestor = b;
		while(possibleAncestor.parentIndex >= 0){
			bAncestorIndices.addFirst(possibleAncestor.sentenceIndex);
			possibleAncestor = possibleAncestor.getParent();
		}

		int ancestorIndex = 0;
		while(aAncestorIndices.peekFirst().equals(bAncestorIndices.peekFirst())){
			ancestorIndex = aAncestorIndices.removeFirst();
			bAncestorIndices.removeFirst();
		}

		return a.getSentenceTokens().get(ancestorIndex);

	}

	public static int ancestorPathLength(Token a, Token ancestor){
		if (a.getSentenceTokens() != ancestor.getSentenceTokens())
			return -1;

		return ancestorPath(a, ancestor).size() - 1;
	}

	public static Deque<Token> ancestorPath(Token a,
			Token ancestor) {
		if (a.getSentenceTokens() != ancestor.getSentenceTokens())
			return null;

		Deque<Token> path = new ArrayDeque<Token>();
		Token currentToken = a;
		path.add(a);

		while(!currentToken.equals(ancestor)){
			currentToken = currentToken.getParent();
			if(currentToken == null)
				return null;
			path.add(currentToken);
		}

		return path;

	}

	public static int dependencyPathLength(Token a,
			Token b){

		if (a.getSentenceTokens() != b.getSentenceTokens())
			return -1;

		return dependencyPath(a, b).size() - 1;

	}

	public static Deque<Token> dependencyPath(Token a,
			Token b){

		if (a.getSentenceTokens() != b.getSentenceTokens())
			return null;

		Token ancestor = getCommonAncestor(a, b);

		Deque<Token> pathA = ancestorPath(a, ancestor);
		Deque<Token> pathB = ancestorPath(b, ancestor);

		pathB.removeLast(); //ancestor is last thing in both pathA and pathB
		while(!pathB.isEmpty())
			pathA.addLast(pathB.removeLast());

		return pathA;
	}
}
